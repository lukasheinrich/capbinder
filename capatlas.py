import pyhf
import numpy as np

def plot_lhood(pdf, obs_data, ax=None, order=None, par_name_dict = None, **par_settings):
    pars = pyhf.tensorlib.astensor(pdf.config.suggested_init())
    for k,v in par_settings.items():
        pars[par_name_dict[k]] = v

    mc_counts = get_mc_counts(pdf,pars)
    bottom = None
    # nb: bar_data[0] because evaluating only one parset
    order = [pdf.config.samples.index(o) for o in (order or pdf.config.samples)]
    for i,sample_index in enumerate(order):
        data = mc_counts[sample_index][0]
        x = np.arange(len(data))
        ax.bar(x, data, 1, bottom = bottom, alpha = 1.0)
        bottom = data if i==0 else bottom + data
    ax.scatter(x, obs_data, c = 'k', alpha = 1., zorder=99)

def get_mc_counts(pdf, pars):
    deltas, factors = pdf._modifications(pars)
    allsum = pyhf.tensorlib.concatenate(deltas + [pyhf.tensorlib.astensor(pdf.thenom)])
    nom_plus_delta = pyhf.tensorlib.sum(allsum,axis=0)
    nom_plus_delta = pyhf.tensorlib.reshape(nom_plus_delta,(1,)+pyhf.tensorlib.shape(nom_plus_delta))
    allfac = pyhf.tensorlib.concatenate(factors + [nom_plus_delta])
    return pyhf.tensorlib.product(allfac,axis=0)

def read_wspace(wspace):
    spec = {
        'channels': wspace['channels'],
        'parameters': wspace['toplvl']['measurements'][0]['config'].get(
            'parameters', []
        ),
    }

    pdf = pyhf.Model(spec, poiname = 'SigXsecOverSM')

    obs_data = wspace['data']['channel1']
    par_selector = {k: v['slice'].start for k,v in pdf.config.par_map.items()}
    nominal = pdf.config.suggested_init()
    nominal_settings = {k: nominal[v] for k,v in par_name_dict.items()}
    order = ['qcd','mc1','mc2','signal']
    return pdf, obs_data, par_selector, nominal_settings, order