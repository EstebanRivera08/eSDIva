%  Make imaging of phantom data for multiple transmit foci
%  and multiple receive foci with apodisation and constant
%  f-number, transmit 4 and receive 2
%
%  Version 1.0, April 7, 1997, Peter Munk
%  Version 1.2, August 14, 1998, Jørgen Arendt Jensen
%       Problem with focusing reference fixed
%**********************************************************
%                      TRANSMIT
%**********************************************************

%  The different focal zones for transmit

xmit_focal_zones_center=[10 20 40 80]'/1000; % focuspoints
xmit_focal_zones=[0 15 30 60]'/1000; % start of zone [m]
Nft=max(size(xmit_focal_zones));
focus_times_xmit=xmit_focal_zones/1540;


%  Set a Hanning apodization on the receive aperture
%  Dynamic opening aperture is used.

Fnumber_xmit=4.0;
xmit_N_active_dyn=round(xmit_focal_zones_center./(Fnumber_xmit*(width+kerf)));

for ii=1:Nft
 if xmit_N_active_dyn(ii)>xmit_N_active 
   xmit_N_active_dyn(ii)=xmit_N_active; end
 xmit_N_pre_dyn(ii) = ceil(xmit_N_active/2  - xmit_N_active_dyn(ii)/2);
 xmit_N_post_dyn(ii) = xmit_N_active - xmit_N_pre_dyn(ii) - xmit_N_active_dyn(ii);
 xmit_apo=(hanning(xmit_N_active_dyn(ii)))';
 xmit_apo_matrix_sub(ii,:)=[zeros(1,xmit_N_pre_dyn(ii)) xmit_apo zeros(1,xmit_N_post_dyn(ii))];
end
apo_zero=(N_elements-xmit_N_active)/2;
xmit_apo_matrix=[zeros(Nft,apo_zero) xmit_apo_matrix_sub zeros(Nft,apo_zero)];

%**********************************************************
%                      RECEIVE
%**********************************************************

%  The different focal zones for receive

%  Set the different focal zones for reception

rec_zone_start=30/1000;
rec_zone_stop=150/1000;
rec_zone_size=20/1000;

rec_focal_zones_center=[rec_zone_start:rec_zone_size:rec_zone_stop]';
rec_focal_zones=rec_focal_zones_center-0.5*rec_zone_size;
Nfr=max(size(rec_focal_zones));
focus_times_rec=rec_focal_zones/1540;


%  Set a Hanning apodization on the receive aperture
%  Dynamic opening aperture is used.

Fnumber_rec=2.0;
rec_N_active_dyn=round(rec_focal_zones_center./(Fnumber_rec*(width+kerf)));

for ii=1:Nfr
 if rec_N_active_dyn(ii)>rec_N_active 
   rec_N_active_dyn(ii)=rec_N_active; end
 rec_N_pre_dyn(ii) = ceil(rec_N_active/2  - rec_N_active_dyn(ii)/2);
 rec_N_post_dyn(ii) = rec_N_active - rec_N_pre_dyn(ii) - rec_N_active_dyn(ii);
 rec_apo=(hanning(rec_N_active_dyn(ii)))';
 rec_apo_matrix_sub(ii,:)=[zeros(1,rec_N_pre_dyn(ii)) rec_apo zeros(1,rec_N_post_dyn(ii))];
end
apo_zero=(N_elements-rec_N_active)/2;
rec_apo_matrix=[zeros(Nfr,apo_zero) rec_apo_matrix_sub zeros(Nfr,apo_zero)];

%**********************************************************
%                      MAKE IMAGE
%**********************************************************

x= -image_width/2;
image_data=0;
for i=1:no_lines
 % Set the focus for this direction (aperture times points(x,y,z))

  xdc_center_focus (emit_aperture,[x 0 0]);
  xdc_focus (emit_aperture, focus_times_xmit, [x*ones(Nft,1), zeros(Nft,1), xmit_focal_zones_center]);
  xdc_center_focus (receive_aperture,[x 0 0]);
  xdc_focus (receive_aperture, focus_times_rec, [x*ones(Nfr,1), zeros(Nfr,1), rec_focal_zones_center]);

   %  Calculate the apodization 

  xdc_apodization (emit_aperture, focus_times_xmit, xmit_apo_matrix);
  xdc_apodization (receive_aperture, focus_times_rec , rec_apo_matrix);

  %   Calculate the received response

  [v, t1]=calc_scat(emit_aperture, receive_aperture, phantom_positions, phantom_amplitudes);

  %  Store the result

  image_data(1:max(size(v)),i)=v;
  times(i) = t1;

  %  Steer in another angle

  x = x + d_x;
  end
