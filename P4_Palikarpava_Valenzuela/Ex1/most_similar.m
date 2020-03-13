function [lab] = most_similar(N,clus_locations)
 distances = zeros(length(clus_locations)); 
 min_dist = 100000;
 lab = -1;
 
 for cl=1:size(clus_locations,1)
  v1 = clus_locations(cl,1);
  v2 = clus_locations(cl,2);
  v3 = clus_locations(cl,3);
  % Calculating current distance
  distances(cl) = ((v1 - N(1))^2 + (v2 - N(2))^2 + (v3 - N(3))^2);
  % Updating min value
  if (distances(cl) < min_dist)
      min_dist = distances(cl);
      lab = cl;
  end
end