# VAE

# Dimensional Flow

# [B,C,H,W]

# [64, 3, 32, 32] #Input_Images
# 1.First encoding and flattening  [64, 3, 32, 32] => [64, 32, 16, 16] => [64, 64, 8, 8] => [64,4096]

# 2.Split into mean and variance
# Linear layer [64, 4096] => [64, 16] for mean and log sigma square
# Mathematical Operation: \(\sigma = \exp(0.5 \times \log\sigma^2)\) to get sigma

# 3.Reparameterization techinque using e - epsilon[64,16] and finding out the latent sample
# z = mean + epsilon * sigma   # [64, 16] + ([64, 16] * [64, 16]) => [64, 16]

# 4.Decoder Expansion + Unflattening
# [64, 16] => [64, 4096] Linear Layer
# [64,4096] => change to view it like [64,64,8,8]
# Transpose Convolution operation [64, 64, 8, 8] => [64, 32, 16, 16] => [64, 3, 32, 32] 


# Now using the Dimensional flow we need to implement the VAE
# simple First perform encoding , using a linear layer find mean and variance and same a z latent vector - using linear layer project it to 4096 and use transpose Convolution and get the 
# final reconstructed Image

# The loss function = Reconstruction loss -> Cross Entropy loss + KL divergence N(0,1) -> N(mu,sigma) => (mu-0)^2 + (sigma-1)^2 => mu^2 + sigma^2 +1 -2sigma
# 																															  => 1/2( mu^2 + sigma^2 -1 -2log(sigma))
																															  
																															  
																															  

class VAEconfig():
	batches = 64
	channels = 3
	height = 32
	width = 32
	latent_dim = 16
	
class VAE(nn.Module):
	def __init__(self,config):
		super().__init__()
		
		#encoder
		self.encode_1 = nn.Conv2d(config.channels,32,kernel_size = 4,stride = 2,padding=1)
		self.encode_2 = nn.Conv2d(32,64,kernel_size = 4 , stride = 2,padding=1)
		#Middle mean and variance
		self.mean = nn.Linear(4096,config.latent_dim)
		self.log_variance = nn.Linear(4096,config.latent_dim)
		
		
		#decoder - z is the latent with [B,16] -> [B,4096]
		self.sample = nn.Linear(16,4096)
		self.decode_1 = nn.ConvTranspose2d(64,32,kernel_size=4,stride=2,padding=1)
		self.decode_2 = nn.ConvTranspose2d(32,3,kernel_size=4,stride=2,padding=1)
		
	def forward(self,input_images):
		B,C,H,W = input_images.size()
		
		en = F.relu(self.encode_1(input_images)) #[64,32,16,16]
		en = F.relu(self.encode_2(en)) #[64,64,8,8]
		en = self.flatten(1) #[64,4096]
		
		mu = self.mean(en) #[64,16]
		log_var = self.log_variance(en) #[64,16]
		
		sigma = torch.exp(0.5 * log_var)
		
		epsilon = torch.randn_like(sigma)
		
		self.z = mu + sigma * epsilon
		
		ex_z = F.relu(self.sample(z)) #[64,4096]
		dec = self.view(B,64,8,8)
		dec_1 = F.relu(self.decode_1(dec))
		dec_2 = torch.sigmoid(self.decode_2(dec_1))
		
		return dec_2,mu,log_var
		
		
		
def vae_loss_function(orginal,reconstructed,mu,logvar):
	
	reconstruction_loss = F.binary_cross_entropy(reconstructed,orginal,reduction="sum")
	
	kl_loss = 0.5 * torch.sum( mu**2 + log_var.exp()-1-log_var)
	
	return reconstruction_loss + kl_loss
	
	
def train_vae(model,data_loader,epochs = 10,learning_rate = 1e-3,device="cuda"):
	
	model = model.to(device)
	model.train()
	
	optimizer = Adam(model.parameters(),lr = learning_rate)
	
	for epoch in range(epochs):
		total_epoch_loss = 0
		
		for images in tqdm(data_loader):
			images = images.to(device)
			
			optimizer.zero_grad()
			
			reconstructed , mu , log_var = model(images)
			
			loss = vae_loss_function(images,reconstructed,mu,log_var)
			
			loss.backward()
			
			optimizer.step()
			
			total_epoch_loss += loss.item()
			
		avg_loss = total_epoch_loss/len(data_loader)
		


def main():
	config = VAEconfig()
	
	if torch.cuda.is_available():
		device = "cuda"
		
	transform = transforms.Compose([
		transforms.ToTensor(),
	])
	
	
	train_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=config.batches, 
        shuffle=True, 
        drop_last=True # Keeps shapes perfectly uniform at exactly 64 items per batch
    )
	
	
	model = VAE(config)
	train_vae(model,train_loader)
	
	torch.save(model.state_dict(),"vae_weights)
	
