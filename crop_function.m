function I1 = crop_function(I)
center1 = ceil(size(I,1)/2);
center2 = ceil(size(I,2)/2);
I1 = I(center1-117:center1+116,center2-175:center2+175,:);
end

