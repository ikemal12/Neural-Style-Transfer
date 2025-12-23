# Neural Style Transfer 

This is a PyTorch implementation of neural style transfer based on the paper [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576) by Gatys et al. 
The aim is to transfer the artistic style of one image onto the content of another using a deep CNN (in this case a pre-trained VGG19 network).

Here is an example of the Taj Mahal mixed with the style The Starry Night by Vincent van Gogh:

<div align="center">
    <img src="images/style/starrynight.jpg" alt="Starry Night" width="256"/>
    <img src="images/content/tajmahal.jpg" alt="Taj Mahal" width=256/>
    <img src="results/tajmahal_styled_with_starrynight_20251223-194256/result.jpg" alt="Starry Taj Mahal" width="512"/>
</div>

---

Here are a couple more examples:

<p align="center">
<img src="images/content/man.jpg" width="270px">
<img src="images/style/mosaic.jpg" width="270px">
<img src="results/man_styled_with_mosaic_20251223-222931/result.jpg" width="270px">

<img src="images/content/golden_gate.jpg" width="270px">
<img src="images/style/sunflowers.jpg" width="270px">
<img src="results/golden_gate_styled_with_sunflowers_20251223-230124/result.jpg" width="270px">
</p>

I also plan to optimize this naive implementation according to the paper Perceptual Losses for Real-Time Style Transfer
and Super-Resolution by Johnson et al. and also apply the algorithm to video streams so it can process in real-time (most likely using OpenCV).
