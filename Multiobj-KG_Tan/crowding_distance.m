function CD=crowding_distance(Front,Fit_Value)
l=length(Front); CD=zeros(1,l);
[row,col]=size(Fit_Value);
ISet=[];
for i=1:l
    ISet(i,:)=Fit_Value(Front(i),:);
end

for i=1:col
    [ISet_S,pos]=sortrows(ISet,i);
    Front_S=Front(pos);          %--重新对此进行排序
    fmax=max(ISet_S(:,i));       %--最大值
    fmin=min(ISet_S(:,i));       %--最小值
    for j=1:l
        flag=find(Front==Front_S(j));  %找到Front相对的位置-%
        if j==1 || j==l
            CD(flag)=+inf;
% if l==1
%     CD(flag)=+inf;
% else if j==1
%     CD(flag)=CD(flag)+2*(ISet_S(j+1,i)-ISet_S(j,i))/(fmax-fmin);
% else if j==l
%         CD(flag)=CD(flag)+2*(ISet_S(j,i)-ISet_S(j-1,i))/(fmax-fmin);
        else
            CD(flag)=CD(flag)+(ISet_S(j+1,i)-ISet_S(j-1,i))/(fmax-fmin);
        end
    end
end
end
