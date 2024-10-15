import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import mayavi.mlab as mlab 
from pystriq import striq
import pandas as pd

from seis_analysis import SeismicDataProcessor
from seis_analysis import SeismicAttributes


import numpy as np
import matplotlib.pyplot as plt

def plot_2d(array, range=None, title_x= 'Inline', title_y = 'TWT'):
    plt.figure(figsize=(10, 6))
    if range is not None:
        plt.imshow(array.T, aspect='auto', cmap='seismic', interpolation='nearest', 
                   extent=[range[0], range[1], range[2], range[3]], origin='lower')
    else:
        plt.imshow(array.T, aspect='auto', cmap='seismic', interpolation='nearest', origin='lower')
    
    plt.colorbar(label='Amplitude')
    plt.xlabel(title_x)
    plt.ylabel(title_y)
    plt.gca().invert_yaxis()

    fig = plt.gcf()
    return fig


def run_app():
    app = QApplication(sys.argv)
    st = striq('Seismic Analysis')


    filepath = r"D:\Project\Data Science\Seismic Processing\Seismic data\Z3AMC1986A-1.sgy"
    processor = SeismicDataProcessor(filepath)
    data_range = processor.get_min_max_values()

    st.tabs('Seismic data',["Data Upload", "Basic Visualization", "Attribute Analysis", "3D Visualization"])
    du_col1, du_col2, du_col3 = st.columns_and_tabs(tab_name="Data Upload", num_columns=3)
    bv_col1, bv_col2, bv_col3 = st.columns_and_tabs(tab_name="Basic Visualization", num_columns=3)
    at_col1, at_col2, at_col3, at_col4 = st.columns_and_tabs(tab_name="Attribute Analysis", num_columns=4)
    td_col1, td_col2, td_col3 = st.columns_and_tabs(tab_name="3D Visualization", num_columns=3)

    file_path = st.file_uploader('Upload a File', column_layout=du_col1)

    st.data_table(label='Survey Details', data=data_range,column_layout=du_col2)


    def plot_il(il):
        inline_fig = processor.plot_specific_inline(iline_val=il)
        return inline_fig
    def plot_xl(xl):
        xline_fig = processor.plot_specific_crossline(xline_val=xl)
        return xline_fig
    def plot_t(t):
        t_fig = processor.plot_specific_time_slice(time_val=t)
        return t_fig

    bv_inline = st.slider("Inline Value:",min_value=data_range['min_inline'], max_value=data_range['max_inline'],default_value=data_range['min_inline'], on_change=lambda value: st.matplotlib_figure(fig=plot_il(value), fig_key='k1'), column_layout=bv_col1)
    bv_xline = st.slider("Crossline Value:",min_value=data_range['min_crossline'], max_value=data_range['max_crossline'],default_value=data_range['min_crossline'], on_change=lambda value: st.matplotlib_figure(fig=plot_xl(value), fig_key='k1'), column_layout=bv_col2)
    bv_twt = st.slider("Time Slice Value:",min_value=data_range['min_time'], max_value=data_range['max_time'],default_value=data_range['min_time'], on_change=lambda value: st.matplotlib_figure(fig=plot_t(value), fig_key='k1'), column_layout=bv_col3)
    
    
    cube = processor.get_3d_cube()
    sa = SeismicAttributes(cube, 0.004)

    seis_attr = {
        'None': None,
        "Coherence": sa.coherence,
        "Cosine_similarity": sa.cosine_similarity,
        "Dip": sa.dip,
        "Dip_magnitude": sa.dip_magnitude,
        "Energy_ratio": sa.energy_ratio,
        "Envelope": sa.envelope,
        "Envelope_phase": sa.envelope_phase,
        "Instantaneous_energy": sa.instantaneous_energy,
        "Instantaneous_frequency": sa.instantaneous_frequency,
        "Instantaneous_phase": sa.instantaneous_phase,
        "Variance": sa.variance,
        "Rms_amplitude": sa.rms_amplitude,
        "Operator 1": sa.azimuthal_anisotropy,
        "Operator 2": sa.frequency_shift,
        "Operator 3": sa.gausse_dip_azimuth,
        "Operator 4": sa.horizontal_derivatives,
        "Operator 5": sa.q_factor,
        "Operator 6": sa.vertical_derivative,
        "Operator 7": sa.wavelet_transform,
        "Operator 8": sa.seismic_wavelet,
        "Operator 9": sa.spectral_decomposition
    }


    select_attr = st.selectbox('Select', options=list(seis_attr.keys()), column_layout=at_col1)

    def il_atr(x):
        idx, _ , _ = processor._get_indices(iline_val=x)
        new_cube = cube[idx:idx+2,:,:]
        sa.seismic_cube = new_cube
        a = seis_attr[select_attr()]() if select_attr() not in ['Envelope_phase', 'Operator 3', 'Operator 4'] else seis_attr[select_attr()]()[1]
        fig_out = plot_2d(array=a[0,:,:], range=[data_range['min_crossline'], data_range['max_crossline'], data_range['min_time'], data_range['max_time']])
        return fig_out

    def xl_atr(x):
        _, idx , _ = processor._get_indices(xline_val=x)
        new_cube = cube[:,idx:idx+2,:]
        sa.seismic_cube = new_cube
        a = seis_attr[select_attr()]() if select_attr() not in ['Envelope_phase', 'Operator 3', 'Operator 4'] else seis_attr[select_attr()]()[1]
        fig_out = plot_2d(array=a[:,0,:], range=[data_range['min_inline'], data_range['max_inline'], data_range['min_time'], data_range['max_time']], title_x='Crossline')
        return fig_out

    def t_atr(x):
        _, _, idx = processor._get_indices(time_val=x)
        new_cube = cube[:,:,idx:idx+2]
        sa.seismic_cube = new_cube
        a = seis_attr[select_attr()]() if select_attr() not in ['Envelope_phase', 'Gausse_dip_azimuth', 'Horizontal_derivatives'] else seis_attr[select_attr()]()[1]
        fig_out = plot_2d(array=a[:,:,0], range=[data_range['min_inline'], data_range['max_inline'], data_range['min_crossline'], data_range['max_crossline']], title_x='Inline', title_y='Xline')
        return fig_out


        
    at_inline = st.slider("Inline Value:",min_value=data_range['min_inline'], max_value=data_range['max_inline'],default_value=data_range['min_inline'], on_change=lambda value: st.matplotlib_figure(fig=il_atr(value), fig_key='k1'), column_layout=at_col2)
    at_xline = st.slider("Crossline Value:",min_value=data_range['min_crossline'], max_value=data_range['max_crossline'],default_value=data_range['min_crossline'], on_change=lambda value: st.matplotlib_figure(fig=xl_atr(value), fig_key='k1'), column_layout=at_col3)
    at_twt = st.slider("Time Slice Value:",min_value=data_range['min_time'], max_value=data_range['max_time'],default_value=data_range['min_time'], on_change=lambda value: st.matplotlib_figure(fig=t_atr(value), fig_key='k1'), column_layout=at_col4)




    td_inline1 = st.slider("Min Inline Value:",min_value=data_range['min_inline'], max_value=data_range['max_inline'],default_value=data_range['min_inline'], column_layout=td_col1)
    td_xline1 = st.slider("Min Crossline Value:",min_value=data_range['min_crossline'], max_value=data_range['max_crossline'],default_value=data_range['min_crossline'], column_layout=td_col2)
    td_twt1 = st.slider("Min Time: ",min_value=data_range['min_time'], max_value=data_range['max_time'],default_value=data_range['min_time'], column_layout=td_col3)

    td_inline2 = st.slider("Max Inline Value:",min_value=data_range['min_inline'], max_value=data_range['max_inline'],default_value=data_range['min_inline'], column_layout=td_col1)
    td_xline2 = st.slider("Max Crossline Value:",min_value=data_range['min_crossline'], max_value=data_range['max_crossline'],default_value=data_range['min_crossline'], column_layout=td_col2)
    td_twt2 = st.slider("Max Time: ",min_value=data_range['min_time'], max_value=data_range['max_time'],default_value=data_range['min_time'], column_layout=td_col3)


    select_attr2 = st.selectbox('Select', options=list(seis_attr.keys()), column_layout=td_col1)


    def td_plot():
        idx01, idx02, idx03 = processor._get_indices(iline_val=td_inline1(),xline_val=td_xline1(),time_val=td_twt1())
        idx11, idx12, idx13 = processor._get_indices(iline_val=td_inline2(),xline_val=td_xline2(),time_val=td_twt2())

        if select_attr2() != 'None':
            new_cube = processor.get_3d_cube()[idx01:idx11,idx02:idx12,idx03:idx13]
            sa.seismic_cube = new_cube
            filt_cube = seis_attr[select_attr2()]() if select_attr2() not in ['Envelope_phase', 'Gausse_dip_azimuth', 'Horizontal_derivatives'] else seis_attr[select_attr2()]()[1]
        else:
            filt_cube = cube[idx01:idx11,idx02:idx12,idx03:idx13]

        maya_fig = processor.plot_cube(cube=filt_cube)
        st.mayavi_figure('3D Seismic', maya_fig,fig_key='k1')

    st.button("Submit", on_click=td_plot,column_layout=td_col3)



    st.tight_layout(du_col1,du_col2, du_col3)
    st.tight_layout(bv_col1,bv_col2, bv_col3)
    st.tight_layout(at_col1,at_col2, at_col3,at_col4)
    st.tight_layout(td_col1,td_col2, td_col3)
    st.tight_layout(st.main_layout)








    st.run()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run_app()



