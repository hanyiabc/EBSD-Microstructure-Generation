function out = randomCrop(img, sz)
    nrow = size(img, 1);
    ncol = size(img, 2);
    
    row = randi(nrow - sz);
    col = randi(ncol - sz);
    sz = sz - 1;
    out = imcrop(img, [col row sz sz]);
end