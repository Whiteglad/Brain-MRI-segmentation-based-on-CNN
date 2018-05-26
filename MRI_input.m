function [label,patch]=MRI_input(stride,size)
% get address of the files downloaded from Brainweb website
% http://www.cma.mgh.harvard.edu/ibsr/
% and return matrix representing image and ground truth segmentation
% i is a 3D matrix representing Brainweb image
% q is manual segmented image

%% read raw data
% f=fopen('psubject04_crisp_v.rawb','r');
% train_seg=reshape(fread(f),262,434,362);
% f=fopen('psubject05_crisp_v.rawb','r');
% test_seg=reshape(fread(f),262,434,362);
% f=fopen('subject04_t1w_p4.rawb','r');
% train_img=reshape(fread(f),262,434,362);
% f=fopen('subject05_t1w_p4.rawb','r');
% test_img=reshape(fread(f),262,434,362);

f=fopen('t1_icbm_normal_1mm_pn3_rf20.rawb','r');
train_img=reshape(fread(f),181,217,181);
train_img=(train_img-128)/128;
f=fopen('phantom_1.0mm_normal_crisp.rawb','r');
label_img=reshape(fread(f),181,217,181);
label_img=(label_img==1 ).*1+(label_img==2 | label_img==8).*2+(label_img==3)*3;

%% 
patch=zeros(size,size,1,[]);
count=0;
a=(randperm(17)+2);
for i=1:1:7
    figure(1);
    subplot(3,7,i);
    imshow(uint8(squeeze(train_img(:,a(i+10)*10,:)*128+128)));
    title(['Ô­Í¼Ïñ',num2str(i)]);
end

for j=a
     slice=padarray( squeeze(train_img(:,j*10,:)) , [fix(size/2),fix(size/2)],0,'both');
%    count=0;
%    imshow(uint8(slice));
     for m=(fix(size/2)+1):stride:fix(size/2)+181
         for n=(fix(size/2)+1):stride:fix(size/2)+181
             count=count+1;
             patch(:,:,1,count)=slice(m+1-ceil(size/2):m+fix(size/2) ,n+1-ceil(size/2):n+fix(size/2));
%            patch(:,:,count,(num-1)*10+j-5)=slice(m+1-fix(size/2):m+fix(size/2) ,n+1-fix(size/2):n+fix(size/2));
         end
     end
end

count=0;

for j=a
     slice=padarray( squeeze(label_img(:,j*10,:)) , [fix(size/2),fix(size/2)],0,'both');
%    count=0;
%    imshow(uint8(slice));
     for m=(fix(size/2)+1):stride:fix(size/2)+181
         for n=(fix(size/2)+1):stride:fix(size/2)+181
             count=count+1;
             label(count)=slice(m ,n);
%            patch(:,:,count,(num-1)*10+j-5)=slice(m+1-fix(size/2):m+fix(size/2) ,n+1-fix(size/2):n+fix(size/2));
         end
     end
end
% label_py=(label==1 ).*1+(label==2 | label==8).*2+(label==3)*3;

% label=(label==1 ).*1+(label==2 | label==8).*2+(label==3)*3;

% patch_py=permute(patch,[4,1,2,3]);
% save('label.mat','label_py');
% save('patch.mat','patch_py');





%%

% [trainImages,trainLabels] = digitTrain4DArrayData;
% 
% idx = randperm(size(trainImages,4),1000);
% valImages = trainImages(:,:,:,idx);
% trainImages(:,:,:,idx) = [];
% valLabels = trainLabels(idx);
% trainLabels(idx) = [];



% mask=((q==1) | (q==2) |(q==3) |(q==8));
% q=(q==1 ).*50+(q==2 | q==8).*100+(q==3)*150;
% 
% % i=mask.*i;
% imshow(uint8(squeeze(q(:,190,:))));
% figure,imshow(uint8(squeeze(i(:,190,:))));