clear all,close all,clc;
figure,
for i = 1:8
    path = ['../images/original images/sample0',int2str(i),'.jpg'];
    img = imread(path);
    
    subplot(4,4,i),imshow(img);title('ԭͼ');
    if length(size(img))>2
        rimg = img(:,:,1);  
        gimg = img(:,:,2);  
        bimg = img(:,:,3);  
        resultr = adapthisteq(rimg);  
        resultg = adapthisteq(gimg);  
        resultb = adapthisteq(bimg);  
        result = cat(3, resultr, resultg, resultb); 
        subplot(4,4,8+i),imshow(result);title('CLAHE�����');
        imwrite(result,['clahe0',int2str(i),'.jpg']);
    end
end