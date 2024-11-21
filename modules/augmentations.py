from torchvision.transforms import v2
import torchvision

class AugmGeneric(object):
    def __init__(
        self,
        img_size=(420,420),
        test = False,
        mean = [0.5]*3,
        std = [0.5]*3,
    ):
        self.test = test

        flip_rotate = v2.Compose([
            v2.RandomHorizontalFlip(p=0.3),
            v2.RandomVerticalFlip(p=0.3),
            v2.RandomApply([v2.RandomRotation(degrees=(-180,180))], p=0.25),
        ])

        self.norm = v2.Normalize(mean, std)

        kernel_odd = int(0.1 *img_size[0])
        if kernel_odd%2==0:
            kernel_odd +=1
 
        self.transforms = v2.Compose([
                                    torchvision.transforms.ToTensor(),
                                    v2.RandomResizedCrop(
                                        size=img_size,  
                                        scale=(0.85, 0.99), antialias=True),
                                    v2.RandomApply([v2.GaussianBlur(kernel_size=kernel_odd)], p=0.2),
                                    flip_rotate,
                                ])
        
        self.no_augm = v2.Compose([
                            torchvision.transforms.ToTensor(),
                            v2.Resize(img_size, antialias=True),
                        ])
    
    def __call__(self, img):
        
        if self.test:
            return self.norm(self.no_augm(img))

        return self.norm(self.transforms(img))

        