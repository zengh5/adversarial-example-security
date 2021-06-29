function f = SRM34671(imagename)
X = imread(imagename);
MAP = ones(size(X));
F = maxSRM(X, MAP);
Ss = fieldnames(F);
f = [];
for Sid = 1:length(Ss)
    Fsingle = eval(['F.' Ss{Sid}]);
    f = [f Fsingle];
end
end