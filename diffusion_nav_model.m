function y_out = diffusion_nav_model()
close all;

dbstop if error
set(0,'Defaultlinelinewidth',3.5, 'DefaultlineMarkerSize',12,...
    'DefaultTextFontSize',5, 'DefaultAxesFontSize',18);

global scrsz
scrsz = get( groot, 'Screensize' );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% define
A_array=2;
B=200; %20; 
D=0.1;
delta_t = 1;
t_array=0:delta_t:275;

SAVE_MOVIE = 1;
INCLUDE_AGENT = 1; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% calc
disp('CALC...');
for A_i = 1:length(A_array)
    curr_A = A_array(A_i);
    disp(['A',num2str(curr_A),'...']);
    source_propagation(curr_A,B,D,t_array,SAVE_MOVIE,INCLUDE_AGENT);
end

y_out=0; 

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function source_propagation(A,B,D,t_array,SAVE_MOVIE,INCLUDE_AGENT)
global scrsz

[source_x_array,source_y_array,wind_x,wind_y]=get_exp_data(); 

pathXY = [source_x_array,source_y_array];
stepLengths = sqrt(sum(diff(pathXY,[],1).^2,2));
stepLengths = [0; stepLengths]; % add the starting point
cumulativeLen = cumsum(stepLengths);
finalStepLocs = linspace(0,cumulativeLen(end), 250);
finalPathXY = interp1(cumulativeLen, pathXY, finalStepLocs);
source_x_array = finalPathXY(:,1); source_y_array = finalPathXY(:,2);

vq = linspace(1,length(wind_x),250);
wind_x = interp1(1:length(wind_x),wind_x,vq); wind_x = wind_x';
wind_y = interp1(1:length(wind_y),wind_y,vq); wind_y = wind_y';

delta_x = 0.5;
delta_y = 0.5;

source_x_array = floor(source_x_array./delta_x);
source_y_array = floor(source_y_array./delta_y);

min_x = min(source_x_array) - delta_x*100; max_x = max(source_x_array)+ delta_x*100;
min_y = min(source_y_array) - delta_y*100; max_y = max(source_y_array)+ delta_y*100;


number_of_sources = length(source_x_array(:));

[x,y] = meshgrid(min_x:delta_x:max_x,min_y:delta_y:max_y);

if SAVE_MOVIE==1
    fig_densities=figure('Position',[1 1 scrsz(4)/2.5 scrsz(3)/1.5]);
    loops = length(2:length(t_array));
    clear F
    F(loops) = struct('cdata',[],'colormap',[]);
end

agent_pos_x = 0; 
agent_pos_y = 0;

position_mem_interval = 25; 
prev_agent_pos_x = zeros(1,position_mem_interval); 
prev_agent_pos_y = zeros(1,position_mem_interval);
curr_c_agent_array = zeros(1,position_mem_interval);

theta = 0; 
%sigma02 = pi/20; %pi/2; 
sigma02 = pi; %pi/2; 
agent_v=delta_x*5; %10*delta_x; %2.5; %1*delta_x;
percentage_steps_per_strategy1 = 0.25; %0.5; %(accurate and slow)
elements_per_strat1 = floor(percentage_steps_per_strategy1*length(t_array));
elements_per_strat2 = length(t_array) - elements_per_strat1; 
strategy_array = [ones(1,elements_per_strat1), 2*ones(1,elements_per_strat2)];
strategy_array = strategy_array(randperm(length(strategy_array)));

t_to_start_agent = 20; 
dir_bias_coef = delta_x;   
curr_px = 0;
curr_py = 0;
prev_px = 0; 
prev_py = 0;
prev_theta = 0;
curr_c_agent = 0;

c = zeros(size(x));
for t_i=2:length(t_array) 
    t=t_array(t_i);

    source_activity = zeros(1,number_of_sources); 
    source_activity(1:(t_i-1)) = t_array(1:(t_i-1));
    
    if INCLUDE_AGENT==1 && (t_i>t_to_start_agent)
        
        %find current up the gradient direction: 
        
        [fx,fy] = gradient(c,delta_x,delta_y); 
        tt = (floor(x./delta_x) == floor(agent_pos_x./delta_x)) & (floor(y./delta_y) == floor(agent_pos_y./delta_y));
        indt = find(tt);

        if norm([fx(indt), fy(indt)])>0, %0.01,
            prev_px = curr_px;
            prev_py = curr_py;
            curr_px = fx(indt)/norm([fx(indt), fy(indt)]);
            curr_py = fy(indt)/norm([fx(indt), fy(indt)]);
            curr_c_agent  = c(indt);
        else
            curr_px = 0; 
            curr_py = 0; 
        end
    end
    
    
    if (t_i>1) && SAVE_MOVIE==1
        figure(fig_densities); clf; 
        surf(x,y,c,'EdgeColor','none'); view(2); 
        title(['t=',num2str(t)]); xlim([min_x max_x]); ylim([min_y max_y]);
        daspect([1 1 1]); colorbar; 
        caxis([0 3.5]); 
        %caxis([0 20]); 
        hold on;
        if t_i<=length(wind_y)
            plot_wind_interval = 15; wind_v_scale = 10; 
            v = wind_v_scale.*wind_y(1:plot_wind_interval:t_i); u = wind_v_scale.*wind_x(1:plot_wind_interval:t_i); 
            quiver3(source_x_array(1:plot_wind_interval:t_i),source_y_array(1:plot_wind_interval:t_i),ones(length(u),1).*100,...
                u,v,zeros(length(u),1),'Color','red','LineWidth',2,'MaxHeadSize',0.5, 'AutoScale','Off'); hold on; 
        end
        
        if INCLUDE_AGENT==1 && (t_i>t_to_start_agent)  && SAVE_MOVIE==1
            figure(fig_densities); 
            switch strategy_array(t_i-1)
                case 1
                    color_string = 'magenta';
                case 2
                    color_string = 'green';
                otherwise disp('bad strategy_array val'); stop;
            end
            scatter3(agent_pos_x,agent_pos_y,5000,150,color_string,'filled'); hold on; 
            %quiver3(min_x:delta_x:max_x,min_y:delta_y:max_y,ones(size(fx)).*350,...
            %    fx,fy,ones(size(fx)).*0,'Color','green','LineWidth',1,'MaxHeadSize',1); hold on; 
            %curr_px = fx(indt)/norm([fx(indt), fy(indt)]);
            %curr_py = fy(indt)/norm([fx(indt), fy(indt)]);
            
            %scatter3(x(indt),y(indt),200,150,'red','filled'); hold on; 
            if indt<=length(x(:))
                quiver3(x(indt),y(indt),6000,...
                    10.*curr_px,10.*curr_py,0,'Color','green','LineWidth',1,'MaxHeadSize',1); hold on;
            end
            
            hold on;  
        end
        
        hold off;
        
        pause(0.001);
        F(t_i-1) = getframe(fig_densities);
        pause(0.001);
        
        pause(0.1);
    end
    
    c = zeros(size(x));
    for source_i=1:number_of_sources 
        if source_activity(source_i) > 0
            curr_t = t-source_activity(source_i);
            curr_c =((A/(curr_t^0.5))*exp(-((((x-source_x_array(source_i)-(wind_x(source_i)*curr_t))).^2)+...
                (((y-source_y_array(source_i))-wind_y(source_i)*curr_t).^2))./(4*D*curr_t))).*(0.5.^(curr_t/B));
            c = c+curr_c;
        end
    end
    

        
    prev_agent_pos_x(1) = [];
    prev_agent_pos_x(end+1) = agent_pos_x;
    prev_agent_pos_y(1) = [];
    prev_agent_pos_y(end+1) = agent_pos_y;
    curr_c_agent_array(1) = [];
    curr_c_agent_array(end+1) = curr_c_agent;


    
    if INCLUDE_AGENT==1 && (t_i>t_to_start_agent)
        
        %find current up the gradient direction: 
        %theta_up_gradient = 
        
        distance_from_mem = sqrt((agent_pos_x-prev_agent_pos_x(1)).^2 + (agent_pos_y-prev_agent_pos_y(1)).^2);
        change_conc_from_mem = curr_c_agent-curr_c_agent_array(1); 

        %if curr_c_agent<=1.5,
        
        if strategy_array(t_i-1) == 1;
            
            if distance_from_mem<=2*agent_v && t_i>position_mem_interval
                strategy_array(t_i) = 2;
            else
                strategy_array(t_i) = 1;
            end
            
        else
            if change_conc_from_mem>0.5,
                strategy_array(t_i) = 1;
            else
                strategy_array(t_i) = 2;
            end
        end
        
        switch strategy_array(t_i)
            case 1 %slow and accurate
                agent_pos_x = agent_pos_x + dir_bias_coef.*curr_px;
                agent_pos_y = agent_pos_y + dir_bias_coef.*curr_py;
            case 2 %fast and noisy 
                if strategy_array(t_i-1)==1,
                    prev_theta = atan2(curr_py,curr_px);
                end
                action = floor(rand (1,1)*3)+1;
                
                switch action
                    case 1 %turn clockwise
                        theta=prev_theta+sigma02*rand(1,1);
                    case 2 %turn counter clockwise
                        theta=prev_theta-sigma02*rand(1,1);
                    case 3 %no turn
                        % do nothing
                end
                agent_pos_x = agent_pos_x + agent_v*cos(theta);
                disp(agent_pos_x)
                agent_pos_y = agent_pos_y + agent_v*sin(theta);
            otherwise disp('bad strategy_array val'); stop;
        end
        
    end
    
    
end

if SAVE_MOVIE==1
    v = VideoWriter(['A',num2str(A),'test.mp4'],'MPEG-4');
    open(v);
    writeVideo(v,F);
    close(v);
end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [x,y]=CRW(N,REALIZATIONS,NreS, v, sigma02,...
    x_initial, y_initial, theta_initial)

x = zeros(REALIZATIONS, N);
y = zeros(REALIZATIONS, N);
theta = zeros(REALIZATIONS, N);
x(:,1) = x_initial;
y(:,1) = y_initial;
theta(:,1) = theta_initial;

for REALIZATION_i = 1:REALIZATIONS
    for step_i = 2:N
        if mod(step_i,NreS)==0
            theta(REALIZATION_i,step_i) = theta_initial;
        else
            action = floor(rand (1,1)*3)+1;
            switch action
                case 1 %turn clockwise
                    theta(REALIZATION_i,step_i) = ...
                        theta(REALIZATION_i,step_i - 1)+sigma02*rand(1,1);
                case 2 %turn counter clockwise
                    theta(REALIZATION_i,step_i) = ...
                        theta(REALIZATION_i,step_i - 1)-sigma02*rand(1,1);
                case 3 %no turn
                    theta(REALIZATION_i,step_i) = ...
                        theta(REALIZATION_i,step_i - 1);
            end
        end
        x(REALIZATION_i,step_i) = x(REALIZATION_i,step_i -1) + ...
            v*cos(theta(REALIZATION_i,step_i));
        y(REALIZATION_i,step_i) = y(REALIZATION_i,step_i -1) + ...
            v*sin(theta(REALIZATION_i,step_i));
    end
    
end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [source_x_array,source_y_array,wind_x,wind_y]=get_exp_data()
source_x_array = [ -0.3700
   21.8300
   45.1400
   69.5600
   92.8700
   95.0900
   96.2000
   97.3100
   98.4200
   98.4200
   97.3100
   88.4300
   86.2100
   86.2100
   83.9900
   76.2200
   86.2100
   95.0900
   96.2000
  103.9700
  111.7400
  123.9500
  136.1600
  148.3700
  133.9400
  119.5100
  116.1800
  127.2800
  121.7300
  110.6300
   93.9800]; 

source_y_array = [ -0.7400
    4.8100
    1.4800
   -1.8500
   -8.5100
  -39.5900
  -70.6700
 -101.7500
 -132.8300
 -162.8000
 -172.7900
 -163.9100
 -189.4400
 -220.5200
 -250.4900
 -281.5700
 -309.3200
 -331.5200
 -360.3800
 -390.3500
 -419.2100
 -446.9600
 -473.6000
 -499.1300
 -520.2200
 -542.4200
 -572.3900
 -600.1400
 -629.0000
 -658.9700
 -684.5000]; 


wind_x = [-0.8699
   -0.5033
   -0.5290
   -0.4769
   -0.4086
   -1.4562
    0.3625
    0.4949
    0.3290
    0.4181
    1.1000
    1.1328
   -0.1717
   -1.5743
   -0.9000
   -0.9988
   -0.7270
   -0.0763
    1.6038
    1.1025
    1.6314
    1.6314
    1.2364
    2.0251
    0.0471
    0.1408
    0.5263
    0.7794
   -0.9000
   -0.0942
    0.6743];

wind_y = [0.9661
    0.7461
    0.7281
    0.7632
   -0.8019
    1.0580 
   -0.1690
   -2.1436
   -2.6799
   -3.9781
   -1.9053
   -1.3989
   -0.8835
   -0.8727
   -1.5588
   -1.9602
   -1.0777
   -0.3927
   -0.8172
   -0.6889
   -0.7607
   -0.7607
   -0.4017
    0.8596
    0.8988
   -0.8889
   -1.7213
   -0.4500
   -1.5588 
   -1.7975
   -1.6689]; 


%wind_x = zeros(size(wind_x));
%wind_y = ones(size(wind_y));

wind_x = 0.1.*(rand(size(wind_x))-0.5);
wind_y = 0.1.*(rand(size(wind_y))-0.5);

source_y_array = linspace(0,50,length(wind_x))'; 
source_x_array = 10.*sin(linspace(-pi,pi, length(wind_x))'); 
  
end





