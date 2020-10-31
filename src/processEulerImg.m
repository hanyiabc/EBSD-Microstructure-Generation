function outImg = processEulerImg(img)
    outImg = img;
    for i = 1:20
        outImg = fillMissingData(outImg);
    end
end