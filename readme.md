# dehaze_cgan

use conditional gan network to dehaze

### introduce

the repository refer to 

> DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks


original paper use the network to deblur,in the repository use the network to dehaze

### dataset prpare

use utils/add_haze.py tool to genernate some haze image,and modify train.py ori_image path and haze_image path

### run

```
python train.py
```

the model result save folder checkpoints/

### test

Put the images you want to test into the folder test/

modify demo.py model path to lastest model
```
python demo.py
```









