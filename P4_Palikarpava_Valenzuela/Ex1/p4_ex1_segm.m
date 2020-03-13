close all;
clear all;

% Choose here how many colors you want to turn white
nc = 5000;

fprintf(1,'Reading image..\n');
S=imread('Santorini.png');
nrows=size(S,1);
ncols=size(S,2);
num_pixels=nrows*ncols;
R=S(:,:,1);
G=S(:,:,2);
B=S(:,:,3);
N=zeros(num_pixels,3);
N(:,1)=reshape(R,1,num_pixels);
N(:,2)=reshape(G,1,num_pixels);
N(:,3)=reshape(B,1,num_pixels);
% Working with data cursor
datacursormode on
figure(1)
image(S);
dcm_obj = datacursormode(figure(1));
% Function that does the segmentation when a point at the screen is
% selected
set(dcm_obj,'UpdateFcn',{@myupdatefcn, nc, R, G, B})

function txt = myupdatefcn(~,event_obj, nc, R, G, B)
% Customizes text of data tips
tstart=tic;
pos = get(event_obj,'Position');
txt = {['X: ',num2str(pos(1))],...
       ['Y: ',num2str(pos(2))]};
% Getting cursor position and point color
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
        %dist((i-1)* cols +j, 1) = int32(int32(int32(int32(r) - int32(R(i, j)))^2)+int32(int32(g) - int32(G(i, j))^2)+int32(int32(int32(b) - int32(B(i, j)))^2));
        
        % Calculating distance between colors
        % In dist we save the distance and the coordinates of the point which has this color
        dist((i-1)* cols +j, 1) = (int32(r) - int32(R(i, j)))^2+(int32(g) - int32(G(i, j)))^2+(int32(b) - int32(B(i, j)))^2;
        
        dist((i-1)* cols +j, 2) = i;
        dist((i-1)* cols +j, 3) = j;
    end
end
% Sorting dist by first column value
dist = sortrows(dist);
%disp(dist)
colors = 0;
curr_col = dist(1,1);
curr_ind = 1;
% Searching for all the colors that must be turned white
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

% Creating image
Sc = zeros(rows, cols, 3);
for c=1:cols
 for r=1:rows
  Sc(r,c,1)=R(r,c);
  Sc(r,c,2)=G(r,c);
  Sc(r,c,3)=B(r,c);
 end
end

figure(2)
image(Sc);
telapsed4=toc(tstart);
fprintf(1,'Segmented image created in %f seconds.\n',telapsed4);

end