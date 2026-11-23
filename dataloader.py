from config import *



class dataset(Dataset):
    def __init__(self, cfg):
        if 'crop' in cfg:
            h, w = [int(x) for x in cfg['crop'].split('_')]
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop((h, w))
            ])
        else:
            self.transform = transforms.ToTensor()

        data_dir = os.path.join(cfg['path'], cfg['dataset'])
        self.data = []

        # read video frames
        for frm in sorted(os.listdir(data_dir)):
            img = cv2.imread(os.path.join(data_dir, frm))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = self.transform(img)
            self.data.append(img)           # [3, H, W]
        
        self.video_size = (len(self.data),) + tuple(self.data[0].shape)            # [T, 3, H, W]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            't': index,
            'Igt': self.data[index],
        }