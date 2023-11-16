import torch

class PGD_Attack_Generator():
    def __init__(self,model,epsilon,num_steps,step_size,norm="L-Inf",loss_fn=None,random_initialization=True,targeted=False,clamp_min=0,clamp_max=1):
        self.model=model
        self.epsilon=epsilon
        self.num_steps=num_steps
        self.step_size=step_size
        self.norm=norm
        if loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        self.random_initialization=random_initialization
        self.targeted=targeted
        self.clamp_min=clamp_min
        self.clamp_max=clamp_max

    def __call__(self,x,labels,target_labels=None,mask=None):
        # Early quit
        current_device = next(self.model.parameters()).device
        if self.epsilon == 0:
            x = x.detach().clone() #.to(current_device)
            return x
        # Check if the model is training - if so, indicate that to be the case and disable gradient calculations for params during adv generation
        is_model_training = self.model.training
        if is_model_training:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.eval()
        x = x.detach().clone() 
        labels = labels.detach().clone() 
        if mask is not None:
            mask = mask.detach().clone() 

        if self.targeted:
            target_labels = target_labels.detach().clone() 

        adv_x = x.detach().clone()

        if self.random_initialization:
            if mask is not None:
                adv_x = adv_x + torch.mul(torch.empty_like(adv_x).uniform_(-self.epsilon,self.epsilon),mask)
            else:
                adv_x = adv_x + torch.empty_like(adv_x).uniform_(-self.epsilon,self.epsilon)
            adv_x = torch.clamp(adv_x,min=self.clamp_min,max=self.clamp_max).detach()

        for _ in range(self.num_steps):
            adv_x.requires_grad = True
            model_outputs = self.model(adv_x)

            if self.targeted:
                loss=-self.loss_fn(model_outputs,target_labels)
            else:
                loss=self.loss_fn(model_outputs,labels)

            with torch.no_grad():
                gradients = torch.autograd.grad(loss,adv_x)[0]
                if mask is not None:
                    gradients = torch.mul(gradients,mask)
                adv_x = adv_x.detach() + self.step_size*gradients.sign()
                eta = torch.clamp(adv_x-x,min=-self.epsilon,max=self.epsilon)
                adv_x = torch.clamp(x+eta,min=self.clamp_min,max=self.clamp_max).detach()

        adv_x.requires_grad = False
        # If the model was training before hand, switch back from eval mode to train mode
        if is_model_training:
            self.model.train()
            for param in self.model.parameters():
                param.requires_grad = True
        return adv_x

