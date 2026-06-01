%  Make imaging of phantom data for a multiple transmit focus
%  and multiple receive foci
%
%  Version 1.0, March 25, 1997, Joergen Arendt Jensen
%  Version 1.2, August 14, 1998, Joergen Arendt Jensen
%       Problem with focusing reference fixed

%  The different focal zones for reception and emission

focal_transmit=[10 20 40 80]'/1000;
Nft=max(size(focal_transmit));
focus_times_transmit=([0 15 30 60]'/1000)/1540;

focal_receive=[10:20:150]'/1000;
Nfr=max(size(focal_receive));
focus_times_receive=(focal_receive-10/1000)/1540;

x= -image_width/2;
image_data=0;
for i=1:no_lines
   %   Set the focus for this direction

  xdc_center_focus (emit_aperture,[x 0 0]);
  xdc_focus (emit_aperture, focus_times_transmit, [x*ones(Nft,1), zeros(Nft,1), focal_transmit]);
  xdc_center_focus (receive_aperture,[x 0 0]);
  xdc_focus (receive_aperture, focus_times_receive, [x*ones(Nfr,1), zeros(Nfr,1), focal_receive]);

   %  Calculate the apodization 
   
  xdc_apodization (emit_aperture, 0, hanning(N_elements)');
  xdc_apodization (receive_aperture, 0, hanning(N_elements)');

  %   Calculate the received response

  [v, t1]=calc_scat(emit_aperture, receive_aperture, phantom_positions, phantom_amplitudes);

  %  Store the result

  image_data(1:max(size(v)),i)=v;
  times(i) = t1;

  %  Steer in another angle

  x = x + d_x;
  end
