import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import diag_gaussian_KLD

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class HVAEAC(nn.Module):
    def __init__(self,proposal_net,prior_net, gen_net,latent_nets):
        super().__init__()
        self.proposal_net = proposal_net
        self.gen_net = gen_net
        self.prior_net = prior_net
        self.latent_nets = latent_nets
        
    def make_observed(self, image, mask):
        observed = torch.tensor(image)
        observed[mask.bool()] = 0
        return observed
    
    def generate_proposal_params(self,image,mask):
        full = torch.cat([image,mask],1)
        proposal_params = self.proposal_net(full)
        N = proposal_params.shape[0]
        D = proposal_params.shape[1]
        L = len(self.latent_nets) + 1
        proposal_mu = proposal_params[:,:(D // 2)].view(N,L,D//(2*L))
        proposal_logvar = proposal_params[:,(D // 2):].view(N,L,D//(2*L))
        
        return proposal_mu, proposal_logvar
    
    def generate_prior_params(self,observed,mask):
        hidden = torch.cat([observed,mask],1)
        prior_params = self.prior_net(hidden)
        
        N = prior_params.shape[0]
        D = prior_params.shape[1]
        L = len(self.latent_nets) + 1
        
        prior_mu = prior_params[:,:(D // 2)].view(N,L,D//(2*L))
        prior_logvar = prior_params[:,(D // 2):].view(N,L,D//(2*L))
        
        return prior_mu, prior_logvar
        
    def reparam(self, mu, log_var):
        eps = torch.randn_like(mu)
        return mu + eps*torch.exp(0.5*log_var)
    
    
    def ELBO(self, image,mask,beta=1):
        observed = self.make_observed(image,mask)
        proposal_mu, proposal_logvar = self.generate_proposal_params(image,mask)
        prior_mu, prior_logvar = self.generate_prior_params(observed,mask)
        
        z = self.reparam(proposal_mu, proposal_logvar)
        recon_image = self.gen_net(z[:,-1,:])
        
        RE = (F.mse_loss(recon_image, image,reduction='none')*mask).view(image.shape[0], -1).sum(-1)
        KLD = diag_gaussian_KLD((proposal_mu[:,0,:],proposal_logvar[:,0,:]),
                                (prior_mu[:,0,:],prior_logvar[:,0,:]),dim=1)
        
        for i in range(len(self.latent_nets)):
            params_q = (proposal_mu[:,(i+1),:],proposal_logvar[:,(i+1),:])
            params_zi = self.latent_nets[i](torch.cat([z[:,i,:],
                                            prior_mu[:,(i+1),:],
                                            prior_logvar[:,(i+1),:]],1))
            D = params_zi.shape[1]
            KLD += diag_gaussian_KLD(params_q,
                                     (params_zi[:,:(D//2)],params_zi[:,(D//2):]),dim=1)
        
        elbo = RE + beta*KLD

        return elbo.mean(), RE.mean(), KLD.mean()
    
    def generate_samples(self,image,mask,N):
        with torch.no_grad():
            observed = self.make_observed(image,mask)
            prior_mu, prior_logvar = self.generate_prior_params(observed,mask)
            
            L = len(self.latent_nets)+1
            D = prior_mu.shape[2]
            recon_images = torch.zeros((N,image.shape[1],image.shape[2],image.shape[3])).to(device)
            for i in range(N):
                z = self.reparam(prior_mu[:,0,:], prior_logvar[:,0,:])
                for j in range(L-1):
                    params_zi = self.latent_nets[j](torch.cat([z.view(1,-1),
                                                prior_mu[:,(j+1),:],
                                                prior_logvar[:,(j+1),:]],1))
                    mu = params_zi[:,:D]
                    logvar = params_zi[:,D:]
                    z = self.reparam(mu, logvar)                    

                recon_images[i,:,:,:] = self.gen_net(z)

            return recon_images*mask + image*(1-mask)
            
