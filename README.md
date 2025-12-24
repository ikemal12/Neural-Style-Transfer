# Neural Style Transfer 

This is a PyTorch implementation of neural style transfer based on the paper [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576) by Gatys et al. 
The aim is to transfer the artistic style of one image onto the content of another using a deep CNN (in this case a pre-trained VGG19 network).

Here is an example of the Taj Mahal mixed with The Starry Night by Vincent van Gogh:

<div align="center">
    <img src="images/style/starrynight.jpg" alt="Starry Night" width="256"/>
    <img src="images/content/tajmahal.jpg" alt="Taj Mahal" width=256/>
    <img src="results/tajmahal_styled_with_starrynight_20251223-194256/result.jpg" alt="Starry Taj Mahal" width="512"/>
</div>

---

Here are a couple more examples:

<p align="center">
<img src="results/pytorch-pretrained-models/candy_man.jpg" width="270px">
<img src="results/pytorch-pretrained-models/rain_man.jpg" width="270px">
<img src="results/man_styled_with_mosaic_20251223-222931/result.jpg" width="270px">

<img src="results/pytorch-pretrained-models/candy_taj_mahal.jpg" width="270px">
<img src="results/tajmahal_styled_with_rain-princess_20251224-190621/result.jpg" width="270px">
<img src="results/pytorch-pretrained-models/mosaic_taj_mahal.jpg" width="270px">
</p>

---

And here are some results coupled with their style:

<p align="center">
<img src="results/gray_bridge_styled_with_vg_la_cafe_20251224-181051/result.jpg" height="267px">
<img src="images/style/vg_la_cafe.jpg" height="267px">
<br><br>
    
<img src="results/gray_bridge_styled_with_wave_crop_20251224-181717/result.jpg" height="267px">
<img src="images/style/wave_crop.jpg" height="267px">
<br><br>

<img src="results/pytorch-pretrained-models/rain_robot.jpg" height="300px">
<img src="images/style/rain-princess.jpg" height="300px">
<br><br>

<img src="results/golden_gate_styled_with_sunflowers_20251223-230124/result.jpg" height="300px">
<img src="images/style/sunflowers.jpg" height="300px">
<br><br>

<img src="results/ronaldo_styled_with_ben_giles_20251223-225458/result.jpg" height="300px">
<img src="images/style/ben_giles.jpg" height="300px">
</p>

---

I have also optimized this naive implementation following [Perceptual Losses for Real-Time Style Transfer
and Super-Resolution](https://arxiv.org/pdf/1603.08155) by Johnson et al., achieving significantly faster inference which enables the algorithm to be applied to videos too!

<p align="center">
    <img src="gifs/monkey.gif" width="250" title="Monkey">
    <img src="gifs/swans.gif" width="250" title="Swans">
    <img src="gifs/tiger.gif" width="250" title="Tiger">
    <br><br>
    <img src="gifs/monkey_candy.gif" width="250" title="Candy monkey">
    <img src="gifs/swans_mosaic.gif" width="250" title="Mosaic swans">
    <img src="gifs/tiger_rain_princess.gif" width="250" title="Rain princess tiger">
</p>
