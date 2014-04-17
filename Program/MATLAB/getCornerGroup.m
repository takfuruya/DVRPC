function [cornerGroups, A] = getCornerGroup(connectedness, thresh, cornersToIgnore)

%[~, cornerGroup] = max(connectedness);

A = connectedness > thresh;
nNodes = size(A, 1);

A(cornersToIgnore, :) = 0;
A(:, cornersToIgnore) = 0;

% Do connected components on adjacency matrix.
A(1:(nNodes+1):end) = 1; % Make diagonals 1.
[nodes, ~, groups, ~] = dmperm(A);
% nodes(groups(1):groups(2)-1) ---- First connected component
% nodes(groups(2):groups(3)-1) ---- Second connected component
% ...
nGroups = numel(groups) - 1;

% Organize data.
cornerGroups = zeros(nNodes, 1);
for i = 1 : nGroups
    cornerGroups(nodes(groups(i):groups(i+1)-1)) = i;
end


end