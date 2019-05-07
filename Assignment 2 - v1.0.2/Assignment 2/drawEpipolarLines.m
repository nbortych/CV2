function drawEpipolarLines(F, im)
    button = 1;
    while button == 1
        [x,y, button] = ginput(1);
        line = F * [x;y;1];

        a = line(1);
        b = line(2);
        c = line(3);

        width = size(im,2);
        x = 0:0.01:width;
        y = (-a*x - c)/b;

        plot(x,y);
    end
end