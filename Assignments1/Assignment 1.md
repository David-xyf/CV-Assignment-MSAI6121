# Assignment 1

## 1. Implement and apply the HE algorithm

### 1.1 Introduce the image histogram

- It is a graphical representation of the intensity distribution of an image.
- It quantifies the number of pixels for each intensity value considered.

![](F:\CV\assignments\Assignments1\introHist.png)

> The picture is from https://docs.opencv.org/master/d4/d1b/tutorial_histogram_equalization.html

### 1.2 Introduce the histogram equalization

- It is a method that improves the contrast in an image, in order to stretch out the intensity range.

- To make it clearer, from the image above, you can see that the pixels seem clustered around the middle of the available range of intensities. What Histogram Equalization does is to *stretch out* this range. Take a look at the figure below: The green circles indicate the *underpopulated* intensities. After applying the equalization, we get an histogram like the figure in the center. The resulting image is shown in the picture at right. 

  > Wikipedia contributors. (2020, September 7). Histogram equalization. In *Wikipedia, The Free Encyclopedia*. Retrieved 02:56, September 18, 2020, from https://en.wikipedia.org/w/index.php?title=Histogram_equalization&oldid=977182912

![](F:\CV\assignments\Assignments1\introHE.png)

> The picture is from https://docs.opencv.org/master/d4/d1b/tutorial_histogram_equalization.html

### 1.3 Explain the HE algorithm

To simplify the procedure of math operation and ignore the complex mathematic formulas, we just show the final implementation of HE formula.

For the discrete situation, we have the HE formula:
$$
s_k=T(r_k)=(L-1)\sum_{j=0}^kp_r(r_j)
\\=\frac{L-1}{MN}\sum_{j=0}^kn_j,k=0,1,2,...,L-1
$$
$MN$ is  the image size, $n_k$ is the number of pixels at the gray level '$r_k$', $L$ is the number of bins (in this assignment, $L$ = 256). $r_k$ is a gray value from an image 

For example, we can imagine the gray distribution of 64x64 pixels, 3-bit image ($L$ = 8) below:

| $r_k$ | $n_k$ | $p_k(r_k)=n_k/MN$ | $s_k$ | $around(s_k)$ |
| :---: | :---: | :---------------: | :---: | :-----------: |
|   0   |  790  |       0.19        | 1.33  |       1       |
|   1   | 1023  |       0.25        | 3.08  |       3       |
|   2   |  850  |       0.21        | 4.55  |       5       |
|   3   |  656  |       0.16        | 5.67  |       6       |
|   4   |  329  |       0.08        | 6.23  |       6       |
|   5   |  245  |       0.06        | 6.65  |       7       |
|   6   |  122  |       0.03        | 6.86  |       7       |
|   7   |  81   |       0.02        | 7.00  |       7       |

So from data above, we can know the equalization can help stretch the image histogram and the updated image could broaden the range of gray scale and improve the contrast.

### 1.4 Implement the HE algorithm

In order to implement the HE algorithm, I make 3 steps to improve the HE algorithm gradually. 

1. I transform the 8 images from RGB to black-white images and apply the basic HE algorithm to the 8 black-white images. The basic function below:

   ```python
   # needed Python libraries 
   from PIL import Image
   import numpy as np
   
   # histogram equalization algorithm for basic pictures
   def histEqAlgo(imgArr):
       hist, bins = np.histogram(imgArr, 255)
       cdf = np.cumsum(hist)
       cdf = 255 * (cdf / cdf[-1])  # mapping
       res = np.interp(imgArr.flatten(), bins[:-1], cdf)
       res = res.reshape(imgArr.shape)
       return res
   ```

2. Based on the function *histEqAlgo* code, I extract RGB data from 3 channels from the colorful images and use  *histEqAlgo* function in 3 channels' data separately. The improved function below:

   ```python
   # histogram equalization algorithm every channel for Colorful pictures
   def RGB_histEqAlgo2(imgArr):
       imgArr = np.array(imgArr)
       arrR = imgArr[..., 0]
       arrG = imgArr[..., 1]
       arrB = imgArr[..., 2]
   
       resR = histEqAlgo(arrR)
       resG = histEqAlgo(arrG)
       resB = histEqAlgo(arrB)
   
       new_imgArr = np.zeros(imgArr.shape, dtype='uint8')
       new_imgArr[..., 0] = resR
       new_imgArr[..., 1] = resG
       new_imgArr[..., 2] = resB
   
       return Image.fromarray(new_imgArr, mode='RGB')
   ```

3. After using the function *RGB_histEqAlgo2* code, I find that the images handled  are a little distorted (the images will be shown in 5). With thought, I draw histograms of the handled images and learn that the new 3 channels' data are separated seriously. So I extract the average of RGB channels as the base to amend the HE algorithm, so as to decrease the distortion in new images. The code below:

   ```python
   def RGB_histEqAlgo3(imgArr):
       imgArr = np.array(imgArr)
       arrR = imgArr[..., 0]
       arrG = imgArr[..., 1]
       arrB = imgArr[..., 2]
   
       # extract the average of RGB data in 3 channels
       imgArr_ave = np.average(imgArr, axis=2)
       hist, bins = np.histogram(imgArr_ave, 255)
       cdf = np.cumsum(hist)
       cdf = 255 * (cdf / cdf[-1])
   
       resR = np.interp(arrR, bins[:-1], cdf)
       resG = np.interp(arrG, bins[:-1], cdf)
       resB = np.interp(arrB, bins[:-1], cdf)
   
       new_imgArr = np.zeros(imgArr.shape, dtype="uint8")
       new_imgArr[..., 0] = resR
       new_imgArr[..., 1] = resG
       new_imgArr[..., 2] = resB
   
       return Image.fromarray(new_imgArr, mode='RGB')
   ```

   

### 1.5 Result of histograms and enhanced images

#### 1.5.1 Show histograms and Compare them

Show histograms of probability-gray pixels and compare the gap between the original samples and samples handled by my implemented algorithm.

<div align="center"><img src="F:\CV\assignments\Assignments1\gray ori vs he.png" style="zoom:80%;"/></div>
<div align="center"><img src="F:\CV\assignments\Assignments1\composed_images_gray_res.jpg"  style="zoom:70%;"/> compare the enhancement (first 8) with original images (last 8)</div>

#### 1.5.2 Apply HE algorithm in RGB channels

<div align="center"><img src="F:\CV\assignments\Assignments1\ori vs HE_1.png"/></div>
<div align="center"><img src="F:\CV\assignments\Assignments1\images_compose.jpg"  />8 original images</div>


<div align="center"><img src="F:\CV\assignments\Assignments1\RGB3channel_res.jpg"  />8 images by function 'RGB_histEqAlgo2'</div>
#### 1.5.3 Apply improved HE algorithm in RGB channels

<div align="center"><img src="F:\CV\assignments\Assignments1\ori vs HE_2.png"/</div>
<div align="center"><img src="F:\CV\assignments\Assignments1\images_compose.jpg" />8 original images</div>

<div align="center"><img src="F:\CV\assignments\Assignments1\images_compose_res.jpg"/>8 images by function 'RGB_histEqAlgo3'</div>
## 2.Discuss features of HE algorithm

#### 2.1 the pro and con of HE algorithm

##### advantages:

- easy to program and apply

- fast to process and cumulate

- HE can be invertible. If the histogram equalization function is known, then the original histogram can be recovered

  > Garg P, Jain T. A comparative study on histogram equalization and cumulative histogram equalization[J]. International Journal of New Technology and Research, 2017, 3(9).

##### disadvantages:

- indiscriminate and information loss.

- the signal gets distorted while increasing the contrast of its background 

  > Krishna A S, Rao G S, Sravya M. Contrast enhancement techniques using histogram equalization methods on color images with poor lightning[J]. International journal of computer science, engineering and applications, 2013, 3(4): 15.

- not take the mean brightness of an image into account

#### 2.2 possible causes of some unsatisfactory contrast enhancement

- **decreasing the valuable signal but increasing the contrast of background noise.** Because from 8 samples, especially in brighter images,  there are more noise among these images,  so that probability of noise is higher than other signals. After using HE algorithm, lots of noise influence the quality of images.
- **cannot achieve equal histogram in reality.** Just seeing the histogram above, we can find that the new histogram can hardly be equal, but in theory, we hope that we could get equalization of contrast through HE algorithm. Actually, HE algorithm cannot improve the contrast perfectly.

Above all, through I read some papers and practiced handling on 8 images, I think that these two causes are mainly possible to produce some unsatisfactory contrast enhancement.



## 3 possible improvements of the basic HE algorithm

### 3.1 abstract

Seeing the images and histograms above, we can find the if there is a large part of light area or dark area in an image, the result by the HE algorithm is not good. So we search a better way to deal with this situation.

**Adaptive histogram equalization** (AHE)  is therefore suitable for improving the local contrast and enhancing the definitions of edges in each region of an image. It differs from ordinary histogram equalization in the respect that the adaptive method computes several histograms, each corresponding to a distinct section of the image, and uses them to redistribute the lightness values of the image. 

However, AHE has a tendency to overamplify noise in relatively homogeneous regions of an image. A variant of adaptive histogram equalization called **contrast limited adaptive histogram equalization** (CLAHE) prevents this by limiting the amplification.

> Wikipedia contributors. (2020, January 21). Adaptive histogram equalization. In *Wikipedia, The Free Encyclopedia*. Retrieved 01:27, September 18, 2020, from https://en.wikipedia.org/w/index.php?title=Adaptive_histogram_equalization&oldid=936814673

Above all, I choose the improved HE algorithm 'CLAHE' to process the samples images.

### 3.2 relevant code

Because of lack of powerful skills and ability of programming, I find it hard to implement CLAHE completely by myself from bottom level to top level. 

Finally, I tend to use 'CLAHE' function in MATLAB from the library '*Image Processing Tool*', because I find that using function 'createCLAHE()' from cv2 in python shows worse result of processing images than same function in MATLAB although same parameters' values, such as clipLimit and tileGridSize, are chosen in two kinds of function.

```matlab
clear all,close all,clc;
figure,
for i = 1:8
    path = ['../images/original images/sample0',int2str(i),'.jpg'];
    img = imread(path);
    
    subplot(4,4,i),imshow(img);title('原图');
    if length(size(img))>2
        rimg = img(:,:,1);
        gimg = img(:,:,2);
        bimg = img(:,:,3);  
        resultr = adapthisteq(rimg);  
        resultg = adapthisteq(gimg);  
        resultb = adapthisteq(bimg);  
        result = cat(3, resultr, resultg, resultb); 
        subplot(4,4,8+i),imshow(result);title('CLAHE处理后');
        imwrite(result,['clahe0',int2str(i),'.jpg']);
    end
end
```

### 3.3 Conclusion and Analysis

#### Results by CLAHE



<div align="center"><img src="F:\CV\assignments\Assignments1\clahe_res.jpg" />enhanced samples by CLAHE</div>
<div align="center"><img src="F:\CV\assignments\Assignments1\CLAHE_samples_histogram.jpg" /></div><div align="center">histogram of 8 enhanced samples by CLAHE</div>

#### Comparison between 3 kinds of histograms and images

<div align="center"><img src="F:\CV\assignments\Assignments1\Conclusion_histogram.png" /></div>
<div align="center"><img src="F:\CV\assignments\Assignments1\comparision.jpg" />original images(last 8) samples by HE(medium 8) samples by CLAHE(last 8)</div>
From above comparison, we can know that although the enhancements by HE are brighter than these by CLAHE,  there is not too light or too dark area in images by CLAHE and images by CLAHE are more real than the images by HE.
