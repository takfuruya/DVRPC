function im2 = rectifyImage(im, T)

im2 = zeros(400, 400, 3, 'uint8');

for i = 1 : 400
    for j = 1 : 400
        p = T * [i/400; j/400; 1];
        p = round(p / p(3));
        px = p(1);
        py = p(2);
        if px > 0 && px <= 640 && py > 0 && py <= 480
            im2(j, i, :) = im(py, px, :);
        end
    end
end

end