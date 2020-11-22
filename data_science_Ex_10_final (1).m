

%% Loading the data 
train_data=readcell('train.csv');
unique=readcell('unique_m.csv');
train_features=cell2mat(train_data(2:end,1:81)); % train features
train_ct=cell2mat(train_data(2:end,82)); % train labels
unique_atoms=cell2mat(unique(2:end,1:86));
unique_ct=cell2mat(unique(2:end,87));

%% Element proportion - Fig.3
prop=zeros(1,size(unique_atoms,2));
for i=1:size(unique_atoms,1)
    temp=zeros(1,size(unique_atoms,2));
    idx=find(unique_atoms(i,:)~=0);
    temp(idx)=1;
    prop=prop+temp;
end
prop=prop/size(unique_atoms,1);
[prop,I]=sort(prop,'descend');
figure
scatter(1:86,prop,10,'filled','k')
text(1:86, prop, unique(1,I), 'HorizontalAlignment','center', 'VerticalAlignment','bottom')
ylabel('Element Proportion')

%% Tc values - Fig.4
figure 
histogram(unique_ct,19,'Normalization','pdf')
xlabel('Critical Temperature(K)')
ylabel('PDF')


%% mean & std Tc - Fig.5 & Fig.6
critic_avg=zeros(1,size(unique_atoms,2));
critic_SD=zeros(1,size(unique_atoms,2));
for i=1:size(unique_atoms,2)
    idx=find(unique_atoms(:,i)~=0);
    critic_avg(i)=mean(unique_ct(idx));
    critic_SD(i)=std(unique_ct(idx));
end

critic_avg_org=critic_avg;
critic_SD_org=critic_SD;

critic_avg(isnan(critic_avg))=0;
[critic_avg,I]=sort(critic_avg,'descend');
figure
scatter(1:86,critic_avg,10,'filled','k')
text(1:86,critic_avg, unique(1,I), 'HorizontalAlignment','center', 'VerticalAlignment','bottom')
ylabel('Mean Critical Temperature(K)')

critic_SD(isnan(critic_SD))=0;
[critic_SD,I]=sort(critic_SD,'descend');
figure
scatter(1:86,critic_SD,10,'filled','k')
text(1:86,critic_SD, unique(1,I), 'HorizontalAlignment','center', 'VerticalAlignment','bottom')
ylabel('SD Critical Temperature(K)')

%% SD VS mean - Fig.7 
figure 
subplot(1,2,1)
scatter(critic_avg_org,critic_SD_org,'filled','k')
xlabel('Mean Critical Temperature(K)')
ylabel('SD Critical Temperature(K)')
% figure 
subplot(1,2,2)
scatter(log(critic_avg_org),critic_SD_org,'filled','k')
xlabel('log(Mean Critical Temperature)(K)')
ylabel('SD Critical Temperature(K)')

%% features correlation
features_corr=mean(mean(abs(corrcoef(train_features))));

%% PCA analysis 
[coeff,~]=pca(train_features);
figure
% subplot(1,2,1)
bar(coeff(:,1))
title('first pc - wtd gmean density')
figure 
% subplot(1,2,2)
bar(coeff(:,2)) 
title('second pc - range density')

%% pca reconstruction 
for i=1:length(coeff)
    [~,index]=max(coeff(:,i));
    pca_idx(i)=index;
end
top_20=train_data(1,pca_idx(1:20));
% display(top_20)

N=40;
train_features_new=train_features(:,pca_idx(1:N));
unique_atoms_new=unique_atoms(:,pca_idx(1:N));

%% linear regression on the attributes training set
x={train_features train_features_new};
y=train_ct;

for j=1:2
    for i=1:25
        % splitting the data to training set and test set
        [y_train,indices_train]=datasample(y,round(0.7*length(y)),1,'Replace',false);
        x_train=x{j}(indices_train,:);
        indices_test=setdiff(1:length(y),indices_train);
        x_test=x{j}(indices_test,:);
        y_test=y(indices_test);

        % train & predict
        lin_fit = fitlm(x_train, y_train);
        scores = predict(lin_fit, x_test);
        %rmse
        mse(i)=sqrt(mean((y_test-scores).^2));
    end
    
    % RMSE
    rmse=mean(mse)
    lin_fit.RMSE; %RMSE option 2

    % R^2
    ss_tot=sum((y_test-mean(y_test)).^2);
    ss_res=sum((y_test-scores).^2);
    r2=1-ss_res/ss_tot
     % R^2 option 2
    r2op2=lin_fit.Rsquared.Ordinary

    % plotting the scores versus predicted data
    figure;
    scatter(y_test, scores, 'filled','k');
    ylabel('predicted temperatures');
    xlabel('observed temperatures');
    beaf={'Before' 'After'};
    title([beaf{j} ' ' 'PCA analysis']);
    hold on
    fplot(@(x) x,'r')
    xlim([-10,200])
    ylim([-50,150])
    % display(lin_fit);
    percision = evaluation(y_test, scores)
end

%% linear regression on the chemical formula set
x={unique_atoms unique_atoms_new};
y=unique_ct;

for j=1:2
    for i=1:25
        % splitting the data to training set and test set
        [y_train,indices_train]=datasample(y,round(0.7*length(y)),1,'Replace',false);
        x_train=x{j}(indices_train,:);
        indices_test=setdiff(1:length(y),indices_train);
        x_test=x{j}(indices_test,:);
        y_test=y(indices_test);

        % train & predict
        lin_fit = fitlm(x_train, y_train);
        scores = predict(lin_fit, x_test);
        mse(i)=mean((y_test-scores).^2);
    end
    
    % RMSE
    rmse=sqrt(mean(mse))
    lin_fit.RMSE; %RMSE option 2
   
    % R^2
    ss_tot=sum((y_test-mean(y_test)).^2);
    ss_res=sum((y_test-scores).^2);
    r2=1-ss_res/ss_tot
    % R^2 option 2
    r2op2=lin_fit.Rsquared.Ordinary
    
    % plotting the scores versus predicted data
    figure;
    scatter(y_test, scores, 'filled','k');
    ylabel('predicted temperatures');
    xlabel('observed temperatures');
    beaf={'Before' 'After'};
    title([beaf{j} ' ' 'PCA analysis']);
    % hold on
    % fplot(@(x) x,'r')
    % xlim([-10,200])
    % ylim([-50,150])
    ylim([-20, 150]);
    % display(lin_fit);
    % the percision we got was 0.3
    percision = evaluation(y_test, scores)
end 

%% deep network for the material properties
performance = zeros(25, 1);
percision = zeros(25, 1);
r2h = zeros(25, 1);
X ={train_features' train_features_new'} ;
y = train_ct';

for j=1:2
    for i = 1:25
        hiddenLayerSize1 = 30; % Number of hidden layers nodes
        hiddenLayerSize2 = 20; % Number of hidden layers nodes
        % trainFcn = 'traingd'; % Gradient descent backpropagation.
        % using adagrad for momentum and variable learning rate
        trainFcn = 'trainbr';
        % net = fitnet([hiddenLayerSize1, hiddenLayerSize2], trainFcn);
        net = feedforwardnet([hiddenLayerSize1, hiddenLayerSize2], trainFcn);
        %net = feedforwardnet([hiddenLayerSize1, hiddenLayerSize2]);
        %view(net);
        net.input.processFcns = {'mapminmax'};
        % the matlab's pca preprocessing function yielded
        % net.input.processFcns = {'processpca'};
        num_samples = size(X{j},2);
        train_percent = 70/100;
        test_percent = 30/100;
        num_train = floor(num_samples*train_percent);
        num_test = floor(num_samples*test_percent);
        random_indecies = randperm(num_samples);
        indecies_train = random_indecies(1:num_train);
        indecies_test = random_indecies(num_train+1:num_train+num_test);
        net.divideFcn = 'divideind';
        net.divideParam.trainInd = indecies_train;
        net.divideParam.testInd = indecies_test;
        net.performFcn = 'mse';
        net.plotFcns = {'plotperform','plottrainstate','ploterrhist', 'plotfit', 'plotroc'};
        % we noticed that with bayesian regulariztion we started to overfit at ~25
        % epochs
        net.trainParam.epochs = 30; % Max number of backpropogation steps
        net.trainParam.min_grad = 1e-10 ; % Minimal training gradient
        net.trainParam.lr = 0.01 ; % Step size
        % using relu as a transfer function of the first layer
        net.layers{1}.transferFcn = 'poslin';
        % using softmax as a transfer function of the second layer
        net.layers{2}.transferFcn = 'softmax';
        [net,tr] = train(net,X{j},y);
        X_test = X{j}(:, indecies_test);
        y_test = y(indecies_test);
        output = net(X_test);
       % using rmse instead of matlabs mse
        performance(i) = sqrt(perform(net,y_test,output));
        % calculating percision by measuring results that were predicted around ~7K
        % from the actual temperature. it seems redundant with the rmse and R^2 but
        % still...
        percision(i) = evaluation(y_test', output');
        % calculating R^2
        ss_tot = sum((y-mean(y)).^2);
        ss_resh1 = sum((y-net(X{j})).^2);
        r2h(i) = 1 - ss_resh1/ss_tot;
    end
    %% testing the network
    average_performance = mean(performance)
    average_percision = mean(percision)
    average_R2 = mean(r2h)
end 

%% deep network for the chemical formulas
performance = zeros(25, 1);
percision = zeros(25, 1);
r2h = zeros(25, 1);
X ={unique_atoms' unique_atoms_new'} ;
y = unique_ct';

for j=1:2
    for i = 1:25
        hiddenLayerSize1 = 30; % Number of hidden layers nodes
        hiddenLayerSize2 = 20; % Number of hidden layers nodes
        % trainFcn = 'traingd'; % Gradient descent backpropagation.
        % using adagrad for momentum and variable learning rate
        trainFcn = 'trainrp';
        net = fitnet([hiddenLayerSize1, hiddenLayerSize2], trainFcn);
        %net = feedforwardnet([hiddenLayerSize1, hiddenLayerSize2], trainFcn);
        %net = feedforwardnet([hiddenLayerSize1, hiddenLayerSize2]);
        % view(net);
        net.input.processFcns = {'mapminmax'};


        num_samples = size(X{j},2);
        train_percent = 70/100;
        test_percent = 30/100;
        num_train = floor(num_samples*train_percent);
        num_test = floor(num_samples*test_percent);
        random_indecies = randperm(num_samples);
        indecies_train = random_indecies(1:num_train);
        indecies_test = random_indecies(num_train+1:num_train+num_test);
        net.divideFcn = 'divideind';
        net.divideParam.trainInd = indecies_train;
        net.divideParam.testInd = indecies_test;
        net.performFcn = 'mse';
        net.plotFcns = {'plotperform','plottrainstate','ploterrhist', 'plotfit', 'plotroc'};
        % we started to overfit at around 2500 epochs
        net.trainParam.epochs = 2600 ; % Max number of backpropogation steps
        net.trainParam.min_grad = 1e-10 ; % Minimal training gradient
        net.trainParam.lr = 0.1 ; % Step size
        % using relu as a transfer function of the first layer
        net.layers{1}.transferFcn = 'poslin';
        % using softmax as a transfer function of the second layer
        net.layers{2}.transferFcn = 'softmax';
        [net,tr] = train(net,X{j},y);
        X_test = X{j}(:, indecies_test);
        y_test = y(indecies_test);
        output = net(X_test);
        % using rmse instead of matlabs mse
        performance(i) = sqrt(perform(net,y_test,output));
        % calculating percision by measuring results that were predicted around ~7K
        % from the actual temperature. it seems redundant with the rmse and R^2 but
        % still...
        percision(i) = evaluation(y_test', output');
        % calculating R^2
        ss_tot = sum((y-mean(y)).^2);
        ss_resh1 = sum((y-net(X{j})).^2);
        r2h(i) = 1 - ss_resh1/ss_tot;
    end
    %% testing the network
    average_performance = mean(performance)
    average_precision = mean(percision)
    average_R2 = mean(r2h)
end

%% Elbow method
num_cells=length(unique_ct);
for k=2:100  
    [idx,C,sumd] = kmeans(unique_atoms,k,'Replicates',20);  
    J(k-1)=sum(sumd)/num_cells;% sumd is already the squared euclidean distance 
end
figure
plot(2:100,J,'-O','LineWidth',1.5,'MarkerSize',10)
title('Elbow plot')

%% Clustering 
for i = 1:1000
k=65;
[idx,C,sumd] = kmeans(unique_atoms,k,'Replicates',10);
% kmeans
for i=1:k
    avg_ct(i)=mean(unique_ct(idx==i));
end
[M,I]=max(avg_ct); %% sometimes Tc too small
a=unique_atoms(idx==I,:);
if M > 103
 break
end
end

%% plot 
for i=1:size(a,1)
    for j=1:size(a,2) 
        if a(i,j)~= 0
            a(i,j)=1;
        end
    end
end

c=sum(a);
[c,I]=sort(c,'descend');
figure
scatter(1:86,c,10,'filled','k')
text(1:86, c, unique(1,I), 'HorizontalAlignment','center', 'VerticalAlignment','bottom')
ylabel('Element Proportion')
