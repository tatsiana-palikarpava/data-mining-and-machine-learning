close all;
clear all;

% Choose here how many clusters (colors) we want to retain
num_clusters=100;
nc = 3;

fprintf(1,'Reading image..\n');
S=imread('Santorini.png');
nrows=size(S,1);
ncols=size(S,2);
num_pixels=nrows*ncols;
R=S(:,:,1);
G=S(:,:,2);
B=S(:,:,3);

image(S);

% The k-NN algorithm accepts as input a matrix N with as many rows as examples (in this case, pixels)
% and as many columns as dimensions of the input space (in this case, three: the color components R, G and B) 
fprintf(1,'Converting image to the requested input format...\n');
N=zeros(num_pixels,3);
N(:,1)=reshape(R,1,num_pixels);
N(:,2)=reshape(G,1,num_pixels);
N(:,3)=reshape(B,1,num_pixels);

% Measuring times in Matlab:
% Store the current clock value in a variable, let's say tstart, by calling the tic funcion.
% Later, the function toc with tstart as an argument returns the time in seconds between the calls to tic and toc.
fprintf(1,'Executing the clustering algorithm...\n');
tstart=tic;
[clus_indexes,clus_locations]=kmeans(N,num_clusters);
telapsed1=toc(tstart);
fprintf(1,'k-means with %d clusters run in %f seconds.\n',num_clusters,telapsed1);
clus_locations=int16(clus_locations);

% The k-NN algorithm returns two column vectors: the first has as many rows as inputs used to train the algorithm
% and, for each of them, contains an integer number in 1..num_clusters that says to which cluster that point (color) belongs to.
% This is called the label assigned to that point.
% The other column vector has num_cluster rows and num_dimensions columns and contains, in the i-th row, the center of the
% cluster number i.

% The Slab image will be filled with the labels, i.e, the numbers of the clusters each point belongs to.
% That image is not to be represented (the labels are not colors). It is a way to store the result in a comfortable way.
fprintf(1,'Creating the image of labels...\n');
Slab=uint8(zeros(nrows,ncols));
tstart=tic;
npix=1;
for c=1:ncols
 for r=1:nrows
  Slab(r,c)=clus_indexes(npix);
  npix=npix+1;
 end;
end;
telapsed2=toc(tstart);
fprintf(1,'Image of labels created in %f seconds.\n',telapsed2);

% Finally, we build the real, clustered image (Sc): each pixel takes as its value the centre of the cluster to which it belongs.
fprintf(1,'Creating the clustered image...\n');
Sc=S;
tstart=tic;
for c=1:ncols
 for r=1:nrows
  Sc(r,c,:)=clus_locations(Slab(r,c),:);
 end
end
telapsed3=toc(tstart);
fprintf(1,'Clustered image created in %f seconds.\n',telapsed3);

figure;
image(Sc);

figure(2);

image(Sc);
R=Sc(:,:,1);
G=Sc(:,:,2);
B=Sc(:,:,3);
datacursormode on
dcm_obj = datacursormode(figure(2));

set(dcm_obj,'UpdateFcn',{@myupdatefcn, nc, R, G, B})
% The k-NN algorithm accepts as input a matrix N with as many rows as examples (in this case, pixels)
% and as many columns as dimensions of the input space (in this case, three: the color components R, G and B) 


function txt = myupdatefcn(~,event_obj, nc, R, G, B)
% Customizes text of data tips
tstart=tic;
pos = get(event_obj,'Position');
txt = {['X: ',num2str(pos(1))],...
       ['Y: ',num2str(pos(2))]};
x = pos(1);
y = pos(2);
r = R(y, x);
g = G(y, x);
b = B(y, x);
rows = size(R, 1);
cols = size(R, 2);
dist = int32(zeros(rows * cols, 3));
for i = 1:rows
    for j = 1:cols
        if i == 58 && j == 58
            disp(1)
        end
        %dist((i-1)* cols +j, 1) = int32(int32(int32(int32(r) - int32(R(i, j)))^2)+int32(int32(g) - int32(G(i, j))^2)+int32(int32(int32(b) - int32(B(i, j)))^2));
        dist((i-1)* cols +j, 1) = (int32(r) - int32(R(i, j)))^2+(int32(g) - int32(G(i, j)))^2+(int32(b) - int32(B(i, j)))^2;
        
        dist((i-1)* cols +j, 2) = i;
        dist((i-1)* cols +j, 3) = j;
    end
end

dist = sortrows(dist);
%disp(dist)
colors = 0;
curr_col = dist(1,1);
curr_ind = 1;
while (colors < nc) && (curr_ind < rows*cols)
    while curr_col == dist(curr_ind) && (curr_ind < rows*cols)
        curr_ind = curr_ind + 1;
    end
    colors = colors + 1;
    curr_col = dist(curr_ind, 1);
end

for i = 1:curr_ind
    R(dist(i, 2),dist(i, 3)) = 255;
    G(dist(i, 2),dist(i, 3)) = 255;
    B(dist(i, 2),dist(i, 3)) = 255;
end
for i = curr_ind:(rows*cols)
    R(dist(i, 2),dist(i, 3)) = 0;
    G(dist(i, 2),dist(i, 3)) = 0;
    B(dist(i, 2),dist(i, 3)) = 0;
end
Sc = zeros(rows, cols, 3);
for c=1:cols
 for r=1:rows
  Sc(r,c,1)=R(r,c);
  Sc(r,c,2)=G(r,c);
  Sc(r,c,3)=B(r,c);
 end
end

figure(3)
image(Sc);
telapsed4=toc(tstart);
fprintf(1,'Segmented image created in %f seconds.\n',telapsed4);

end