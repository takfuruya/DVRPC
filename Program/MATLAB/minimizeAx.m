function out = minimizeAx(A)
% Find x which minimizes ||A*x|| subject to ||x||=1
% @return x as a row vector

[U, S, V] = svd(A);
out = V(:, end);

end