# Copyright (c) Arnaud Cassan.
# Distributed under the terms of the MIT license.

import sys
import os
import fnmatch
import ftplib
import warnings
import numpy as np
import subprocess
import pandas as pd
from itertools import permutations
from time import gmtime, strftime
import urllib2
from bokeh.layouts import column
import bokeh.plotting as bplt
from bokeh.models import Range1d, Arrow, VeeHead
from bokeh.embed import components
from microlensing.utils import checkandtimeit, verbosity, printi, printd, printw
from microlensing.mismap import version_info
from microlensing.mismap.magclightc import MagnificationCurve, LightCurve
from microlensing.mismap.caustics import Caustics

def create_fitlist():
    """Create list of events to be fitted or not"""
    
    rsync_ARTEMiS_alerts()

    df = pd.read_csv('ARTEMiS_alerts.txt', header=None, skip_blank_lines=True, comment='#', names=['col'])

    # write list of alerts, and update 'K' to 'M' for MOA
    evarte, evfit = '', list()
    for dd in df['col'].values:
        ligne = 'fit_artemis ' + dd.split(' ')[0] + '\n'
        evarte += ligne.replace('K', 'M')
        ligne = dd.split(' ')[0]
        evfit.append(ligne.replace('K', 'M'))

    f = open('addevents.dat', 'r')
    evman = f.read()
    f.close()

    f = open('.allevents.dat', 'r')
    evallold = f.read()
    f.close()

    evadd = ''
    for ev in evman.split('\n')[0:-1]:
        if ev not in evfit:
            evadd += 'fit_add ' + ev + '\n'

    evtofit = np.unique(np.array(evfit + evman.split('\n')[0:-1]))

    evstop = ''
    for ev in evallold.split('\n')[0:-1]:
        if ev not in evtofit:
            evstop += 'stop ' + ev + '\n'

    f = open('eventslist.dat', 'w')
    f.write(evarte + evadd + evstop)
    f.close()

    # save history list of events
    evall = np.unique(np.array(evfit + evman.split('\n')[0:-1] + evallold.split('\n')[0:-1]))

    f = open('.allevents.dat', 'w')
    for ev in evall:
        f.write(ev + '\n')
    f.close()

def rsync_ARTEMiS_data(event, syncdir):
    """Syncrhonize data with ARTEMiS
        
        Returns:
        --------
        0 : synchronization succeeded and
            local data were updated,
        1 : synchronization succeeded but
            local data were already up to date,
        2 : synchronization failed.
        """
    # set I/O shell display
    tcol, tun, tend, tit = "\033[0m\033[34m", "\033[0m\033[1m\033[34m", "\033[0m", "\033[0m\033[3m"
    
    # check event name format
    if len(event) != 8:
        raise ValueError("wrong format for event name: correct is 'xByyzzzz'")

    # get microlensing season
    year = '20' + event[2:4]
    
    #    rsync -azu ML_RsyncNet@mlrsync-stand.net::Data2018/OOB180088I.dat .
    printi(tcol + "Rsync ARTEMiS database" + tend)

    # password in file
    pswd = 'pswd_artemis.txt'

    # archive format: XKB180039I, with X=dataset, I=filter
    artemisevent = event.replace('M', 'K')
    proc = subprocess.Popen('rsync -avzu -L --password-file=' + pswd + ' ML_RsyncNet@mlrsync-stand.net::Data' + year + '/*' + artemisevent + '*.dat ' + syncdir + ' --stats | grep "files transferred:"', shell=True, executable='/bin/bash', stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.wait()
    
    # get std output and error
    stdout, stderr = proc.communicate()
    
    # treat rsync outputs
    if stderr:
        printw(tun + "Failed to rsync ARTEMiS\n" + tit + stderr + tend)
        return 2
    else:
        # check if files were updated
        if ' 0' in stdout:
            printd(tit + "  (files have not changed on ARTEMiS server)" + tend)
            return 1
        else:
            printd(tit + "  (local files have been updated from ATERMiS server)" + tend)
            return 0

def rsync_ARTEMiS_alerts():
    """Syncrhonize alerts with ARTEMiS Signalmen
        
        Returns:
        --------
        0 : download succeeded,
        2 : download failed.
        """
    # set I/O shell display
    tcol, tun, tend, tit = "\033[0m\033[34m", "\033[0m\033[1m\033[34m", "\033[0m", "\033[0m\033[3m"
    
    #    rsync -azu ARTEMiSuser@mlrsync-stand.net::FollowUp/FollowUpZ.signalmen .
    printi(tcol + "Rsync ARTEMiS Signalmen" + tend)
    
    # password in file
    pswd = 'pswd_signalmen.txt'
    # archive format: XKB180039I, with X=dataset, I=filter
    proc = subprocess.Popen('rsync -az --password-file=' + pswd + ' ARTEMiSuser@mlrsync-stand.net::FollowUp/FollowUpZ.signalmen ARTEMiS_alerts.txt', shell=True, executable='/bin/bash', stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.wait()
    
    # test whether data were downloaded, and if not treat as exception
    stdout, stderr = proc.communicate()
    
    if stderr:
        printw(tun + "Failed to rsync ARTEMiS Signalmen\n" + tit + stderr + tend)
        return 1
    else:
        printd(tit + "  (list of alerts downloaded from ATERMiS Signalmen)" + tend)
        return 0

def fill_webpage(inhtml, outhtml, fillwith):
    """Fill the template html page with new html content
        
        Example
        -------
        >>> fillwpage('in.html', 'out.html', ('_keyword_', var))
        >>> fillwpage('in.html', 'out.html', [('_keyword1_', var1), ('_keyword2_', var2)])
        """
    # set I/O shell display
    tcol, tun, tend, tit = "\033[0m\033[33m", "\033[0m\033[1m\033[33m", "\033[0m", "\033[0m\033[3m"
    
    # important - add these lines between <header> tags --> mettre dans le code event.html
    """<link rel="stylesheet" href="http://cdn.pydata.org/bokeh/release/bokeh-0.12.9.min.css" type="text/css" /><script type="text/javascript" src="http://cdn.pydata.org/bokeh/release/bokeh-0.12.9.min.js"></script><!-- _SCRIPT_ -->"""
    
    # fill html page
    with open(inhtml, 'r') as f:
        cont = f.read()
        if type(fillwith) == tuple:
            fillwith = [fillwith]
        for fill in fillwith:
            key, new = fill
            cont = cont.replace(key, new)
        f.close()
    with open(outhtml, 'w') as f:
        f.write(cont)
        f.close()

    # verbose
    printi(tcol + "Fill document : " + tit + "'" + inhtml + "'" + tcol + " >>> " + tit + "'" + outhtml + "'" + tend)
    for fill in fillwith:
        printd(tit + "  (keyword: " + fill[0] + ")" + tend)

def ftp_miiriads(infile, outfile):
    """Send filled html page to miiriads.iap.fr web site"""
    # set I/O shell display
    tcol, tun, tend, tit = "\033[0m\033[34m", "\033[0m\033[1m\033[34m", "\033[0m", "\033[0m\033[3m"
    
    # verbose
    printi(tcol + "Uploading to miiriads.iap.fr " + tit + "'" + infile + "'" + tcol + " >>> " + tit + "'" + outfile + "'" + tend)
    
    # setup ftp connection
    try:
        ftp = ftplib.FTP('webeur2.iap.fr', 'miiriads', '1OrMore!!')
        path, filename = split(outfile)
        ftp.cwd(path)
        ftp.storbinary('STOR {}'.format(filename), open(infile))
        ftp.quit()
    except:
        printw(tun + "Failed to upload '" + tit + infile + tun + "' to miiriads.iap.fr" + tend)

def retrieve(event, crossrefs=None, download=True):
    """Create or update microlensing event data files
        
        Calling
        =======
        retrieve(event)
        
        Parameters
        ----------
        event : string
            Name of microlensing event: 'xByyzzzz', where:
                x is O (OGLE alert) or B (MOA alert)
                yy is the year
                zzzz is the event ID
        Example
        -------
        >>> datasets = retrieve('OB180222')
        """
    # set I/O shell display
    tcol, tun, tend, tit = "\033[0m\033[34m", "\033[0m\033[1m\033[34m", "\033[0m", "\033[0m\033[3m"
    
    # check event name format
    if len(event) != 8:
        raise ValueError("wrong format for event name: correct is 'xByyzzzz'")

    # get microlensing season
    year = '20' + event[2:4]

    # directories arborescence
    datapath = event + '/data/'
    syncdir = datapath + 'sync/'
    fitsdir = event + '/fits/'
    if not os.path.isdir(event):
        # create dir: ./event/
        printi(tcol + "Create new event folder " + tit + event + "/" + tend)
        proc = subprocess.Popen('mkdir ' + event, shell=True, executable='/bin/bash')
        proc.wait()
#    else:
#        printi(tcol + "Update event " + tit + event + tend)
    if not os.path.isdir(datapath):
        # create dir: ./event/data/
        proc = subprocess.Popen('mkdir ' + datapath, shell=True, executable='/bin/bash')
        proc.wait()
    if not os.path.isdir(syncdir):
        # create dir: ./event/data/sync/
        proc = subprocess.Popen('mkdir ' + syncdir, shell=True, executable='/bin/bash')
        proc.wait()
    if not os.path.isdir(fitsdir):
        # create dir: ./event/fits/
        proc = subprocess.Popen('mkdir ' + fitsdir, shell=True, executable='/bin/bash')
        proc.wait()

    # check whether to get new data, loca data or stop
    if download:
        # if crossfrefs file is given
        newkmt = []
        if crossrefs:
            newsync = 3
            refs = getcrossrefs(crossrefs, event)
#            newkmt = []
            for ref in refs.split():
                if 'K' in ref:
                    newkmt = addkmt(event, ref)
                elif newsync != 0:
                    newsync = rsync_ARTEMiS_data(ref, syncdir)
    
        # if crossfrefs file not given:
        else:
            if 'K' in event:
                newkmt = addkmt(event, event)
            else:
                newsync = rsync_ARTEMiS_data(event, syncdir)

        # check if proceeds
        if not (newsync == 0 or newkmt):
            printi(tun + "No new data available for " + tit + event + tun + " (nothing to do)" + tend)
            sys.exit(0)
        if newkmt:
            printi(tun + "New KMT data available for " + tit + event + tend)
        if newsync == 0:
            printi(tun + "New SYNC data available for " + tit + event + tend)
    else:
        printi(tcol + "Use local datasets" + tend)

    # create final list od datasets
#    printi(tcol + "Create final list of datasets" + tend)
#    kevent = event
#    syncset = fnmatch.filter(os.listdir(syncdir), '*' + event.replace('M', 'K') + '*.dat')
    syncset = fnmatch.filter(os.listdir(syncdir), '*.dat')
    for ds in syncset:
        reformat(syncdir + ds, datapath + ds.replace('K', 'M'), ['mag', 'errmag', 'hjd'])
#
#        if 'K' in ds:
#            proc = subprocess.Popen('cp ' + syncdir + ds + ' ' + syncdir + ds.replace('K', 'M'), shell=True, executable='/bin/bash')
#            proc.wait()
#    syncset = [ds.replace('K', 'M') for ds in syncset]

    # fetch other datasets in data/
    othersets = fnmatch.filter(os.listdir(datapath), '*.dat')
    if os.path.isfile(datapath + 'params.dat'):
        othersets.remove('params.dat')

    # remove duplicates
    datasets = list(np.unique(syncset + othersets))

    # discard symblinks (useful only if rsync is used without -L option)
    datasets = [dataset for dataset in datasets if not os.path.islink(syncdir + dataset)]

    # get OGLE first, if not MOA first, otherwise keep order
    datasets = orderlist(datasets)

## reformat only sync/ datasets ## A FAIRE PLUS HAUT !!!
#    for dataseti in datasets:
#        if os.path.isfile(syncdir + dataseti):
#            reformat(syncdir + dataseti, datapath + dataseti, ['mag', 'errmag', 'hjd'])

    # add relative path to datasets names
    for i in range(len(datasets)):
        datasets[i] = datapath + datasets[i]

    printi(tcol + "Found " + tit + str(len(datasets)) + tcol + " dataset(s)" + tend)
    printd(tit + "  " + str(datasets) + tend)

    return datasets

def getcrossrefs(filename, event):
    """Update cross-reference file for events observed by different surveys
        
        Calling
        =======
        getcrossrefs(filename, event)
        
        Parameters
        ----------
        filename : file
            File with cross-reference names of all events
            
        event : string
            Reference survey event name
        
        Example
        -------
        >>> getcrossrefs('crossrefs.dat', 'OB190222')
        >>> 'OB190222 KB190202 MB190234'
        """
    # set I/O shell display
    tcol, tun, tend, tit = "\033[0m\033[34m", "\033[0m\033[1m\033[34m", "\033[0m", "\033[0m\033[3m"
    
    if not os.path.isfile(filename):
        raise IOError("file '" + filename + "' is missing")

    df = pd.read_csv(filename, names=['col'])

    dfmatch = df[df['col'].str.contains(event)]

    eventrefs = list(dfmatch['col'].values)

    if len(eventrefs) == 0:
        return event
    elif len(eventrefs) == 1:
        return ' '.join(eventrefs)
    else:
        return ' '.join([eventrefs[0]])

def addkmt(event, kmtname):
    """Add munally KMTNet data to a new or existing event"""
    # set I/O shell display
    tcol, tun, tend, tit = "\033[0m\033[34m", "\033[0m\033[1m\033[34m", "\033[0m", "\033[0m\033[3m"
    
    # check event name format
    if len(event) != 8 or len(kmtname) != 8:
        raise ValueError("wrong format for event name: correct is 'xByyzzzz'")
    
    printi(tcol + "Try fetch KMT data " + tit + kmtname + tcol + " for event " + tit + event + tend)

    # get microlensing season
    year = '20' + event[2:4]

    # directories arborescence
    datapath = event + '/data/'
    kmtdir = datapath + 'kmt/'
    fitsdir = event + '/fits/'
    if not os.path.isdir(event):
        # create dir: ./event/
        printi(tcol + "Create new event folder " + tit + event + "/" + tend)
        proc = subprocess.Popen('mkdir ' + event, shell=True, executable='/bin/bash')
        proc.wait()
#    else:
#        printi(tcol + "Update event " + tit + event + tend)
    if not os.path.isdir(datapath):
        # create dir: ./event/data/
        proc = subprocess.Popen('mkdir ' + datapath, shell=True, executable='/bin/bash')
        proc.wait()
    if not os.path.isdir(fitsdir):
        # create dir: ./event/fits/
        proc = subprocess.Popen('mkdir ' + fitsdir, shell=True, executable='/bin/bash')
        proc.wait()
    if not os.path.isdir(kmtdir):
        # create dir: ./event/data/kmt/
        proc = subprocess.Popen('mkdir ' + kmtdir, shell=True, executable='/bin/bash')
        proc.wait()

    # get data and uncompress
    url = 'http://kmtnet.kasi.re.kr/ulens/event/' + year + '/data/' + kmtname + '/pysis/pysis.tar.gz'
    remotefile = urllib2.urlopen(url)
    arch = remotefile.read()
    with open(kmtdir + 'pysis.tar.gz', 'wb') as f:
        f.write(arch)
        f.close()

    # check whether data are new : if not, stop and returne False
    if os.path.isfile(kmtdir + 'kmt.tar.gz'):
        if os.path.getsize(kmtdir + 'pysis.tar.gz') == os.path.getsize(kmtdir + 'kmt.tar.gz'):
            return False

    proc = subprocess.Popen('cp ' + kmtdir + 'pysis.tar.gz ' + kmtdir + 'kmt.tar.gz', shell=True, executable='/bin/bash')
    proc.wait()

    # untargz data
    proc = subprocess.Popen('cd ' + kmtdir + ' ; tar -xvzf kmt.tar.gz', shell=True, executable='/bin/bash')
    proc.wait()

    # fetch datasets and reformat data
#    dr = ' ' + kmtdir
#    for obs in ['A', 'C', 'S']:
    listdat = fnmatch.filter(os.listdir(kmtdir), 'KMT*_I.pysis')
    for dat in listdat:
        reformat(kmtdir + dat, datapath + dat[3:6] + event[2:8] + 'I.dat', ['hjd', 'dflux', 'fluxerr', 'mag', 'errmag', 'seeing', 'backg'])


#        com = dr.join(['cat '] + L) + ' > ' + kmtdir + 'tmp.pysis'
#        proc = subprocess.Popen(com, shell=True, executable='/bin/bash')
#        proc.wait()
#        reformat(kmtdir + 'tmp.pysis', datapath + obs + event + 'I.dat', ['hjd', 'dflux', 'fluxerr', 'mag', 'errmag', 'seeing', 'backg'])

    return True

def tmp_OGLE(event):
    # set I/O shell display
    tcol, tun, tend, tit = "\033[0m\033[34m", "\033[0m\033[1m\033[34m", "\033[0m", "\033[0m\033[3m"
    
    # default values
    t0, tE  = 8630., 30.
    
    # check whether OGLE data set exist
    if event[0] == 'O':
        ## attention si MOA ca prend le mauvais data set !
        evnum = event[4:8]
        
        # get microlensing season
        year = '20' + event[2:4]
        
        # ftp OGLE
        try:
            ogledir = '/ogle/ogle4/ews/' + year + '/' + 'blg-' + evnum + '/'
            ftp = ftplib.FTP('ftp.astrouw.edu.pl', 'anonymous', '')
            ftp.cwd(ogledir)
            ftp.retrbinary('RETR params.dat', open('./' + event + '/data/params.dat', 'wb').write)
            ftp.quit()
        
            # get OGLE PSPL best-fit parameters
            df = pd.read_csv('./' + event + '/data/params.dat', names=['col'])
            t0 = float([a for a in df['col'].values[6].split(' ') if a][1]) - 2450000.
            tE = float([a for a in df['col'].values[7].split(' ') if a][1])

            printi(tcol + "OGLE best-fit PSPL parameters " + tit + "(t0={0}, tE={1})".format(t0, tE) + tcol + " found" + tend)
        except:
            printw(tun + "Failed to connect to ftp.astrouw.edu.pl" + tend)
 
    return t0, tE

def reformat(infilename, outfilename, cols, offset=0.):
    """Reformat data files to be used by muLAn
        
        Calling formatdata
        ==================
        reformat(infilename, outfilename, cols)
        
        Usage
        -----
        Enter in cols the list of columns description (i.e. keywords,
        see below) in the order they appear in the input file.
        
        Parameters
        ----------
        infilename: string
            Name of input data file.
        outfilename: string
            Name of output data file in muLAn format.
        cols: sequence of strings
            Mandatory keywords are:
                'hjd': Julian date or modified Julian date.
                'mag': magnitude.
                'errmag': error in magnitude.
            Optional keywords are:
                'seeing': seeing.
                'backg': background.
            For useless columns, use e.g. 'other'

        Examples
        --------
        >>> formatdata('data.dat', 'data_muLAn.dat',
                ['hjd', 'mag', 'errmag', 'seeing', 'backg'])
        
        >>> formatdata('data.dat', 'data_muLAn.dat',
                ['other', 'hjd', 'mag', 'errmag'])
        """
    # set I/O shell display
    tbf, tcol, tend, tit = "\033[0m\033[1m", "\033[0m\033[35m", "\033[0m", "\033[0m\033[3m"
    
    # check mandatory keywords
    mandatkeys = ['hjd', 'mag', 'errmag']
    for key in mandatkeys:
        # check whether all mandatory keywords are present
        if key not in cols:
            raise ValueError("mandatory column {} missing".format(key))
        # check whether keywords appear only once
        if cols.count(key) > 1:
            raise ValueError("column {} appears more than once".format(key))

    # check if input file exists
    if not os.path.isfile(infilename):
        raise IOError("file '" + infilename + "' is missing")
    
    # limit number of columns to read
    usecols = range(len(cols))
    # reading input data file
    dtype = {'names': tuple(cols), 'formats': tuple(['S50' for c in cols])}
    data = np.loadtxt(infilename, dtype=dtype, usecols=usecols, unpack=False)
    # re-order columns
    newfile = ''
    for i in range(len(data['hjd'])):
        # check whether date is in HJD or MHJD, and correct it
        mhjd = float(data['hjd'][i]) - 2450000.
        if mhjd > 0.:
            data['hjd'][i] = str(mhjd)
        # add offset to agnitudes is required
        mag = float(data['mag'][i]) + offset
        data['mag'][i] = str(mag)
        # mandatory keywords
        newfile = newfile + repr(i + 1) + ' ' + data['hjd'][i] + ' ' + data['mag'][i] + ' ' + data['errmag'][i]
        # optional keywords
        if 'seeing' in cols:
            newfile = newfile + ' ' + data['seeing'][i]
        else:
            newfile = newfile + ' 0'
        if 'backg' in cols:
            newfile = newfile + ' ' + data['backg'][i] + '\n'
        else:
            newfile = newfile + ' 0\n'

    # create output data file in muLAn format
    outfile = open(outfilename, 'w')
    outfile.write(newfile)
    outfile.close()
    
    # verbose
    printi(tcol + "Reformat data file " + tit + infilename + tcol + " to " + tit + outfilename + tend)

def makehtml_event(html_event_template, event, models, datasets, trange=None, crossrefs=None):
    """Create individual event html page"""
    
    # set I/O shell display
    tcol, tun, tend, tit = "\033[0m\033[35m", "\033[0m\033[1m\033[35m", "\033[0m", "\033[0m\033[3m"
    
    # verbose
    printi(tcol + "Creating " + tit + "'" + event + '/' + event + '.html' + "'" + tcol + " web page" + tend)
    if crossrefs:
        printi(tcol + "All names of event: " + tit + crossrefs + tend)

    # date and time of event's update
    with open(event + '/lastupdate.txt', 'w') as f:
        fittime = strftime('- Last update: %d/%m/%Y at %H:%M:%S UT', gmtime())
        f.write(fittime)
        f.close()

    # create bokeh column of best-fit models
    mc = MagnificationCurve()
    lc = LightCurve(datasets, trange=trange, dmag=None)
    si = list()
    titlmod = 'Binary-lens model '
    tools = ["pan", "wheel_zoom", "box_zoom", "undo", "redo", "reset", "save"]
    active_drag = "pan"
    active_scroll = "wheel_zoom"

    for i in range(len(models)):
        
        printi(tcol + "Computing " + tit + "model " + str(i + 1) + tend)
        
        # compute best-fit light curve
        mc.create(models[i])
        lc.fit(mc, 'central', '+', init=None)
        
        # bind x-axis on first plot
        if i == 0:
            plot = bplt.figure(width=800, plot_height=400, title=titlmod + '1', tools=tools, active_drag=active_drag, active_scroll=active_scroll)
            si.append(plot)
            if trange:
                tmin = lc.params['t0'] - 1.5 * lc.params['tE']
                tmax = lc.params['t0'] + 1.5 * lc.params['tE']
                si[0].x_range = Range1d(tmin, tmax)
        else:
            plot = bplt.figure(width=800, plot_height=400, title=titlmod + str(i + 1), x_range=si[0].x_range, tools=tools, active_drag=active_drag, active_scroll=active_scroll)
            si.append(plot)

        # compute blended flux
        flux = lc._mu * lc.params['Fs'][0] + lc.params['Fb'][0]
        t = lc._t * lc.params['tE'] + lc.params['t0']

        # remove points with negative flux and NaN
        arg = np.logical_not(np.isnan(flux)) & (flux > 0.)
        flux = flux[arg]
        t = t[arg]

        # invert y-axis
        si[i].y_range = Range1d(1.02 * np.max(-2.5 * np.log10(flux)), 0.98 * np.min(-2.5 * np.log10(flux)))

        # theoretical ligth curve
        si[i].line(t, -2.5 * np.log10(flux), color='darkorange')

        # add data on plot
        k = 0
        atel = iter(color_tel)
        for dat in lc._datalist:
            # color and legend
            _ , datai = split(dat[6])
            tel = datai[0]
            if tel not in color_tel:
                tel = atel.next()
            color = color_tel[tel]
 
            # deblending
            magnif = (np.power(10., - dat[2] / 2.5) - lc.params['Fb'][k]) / lc.params['Fs'][k]
            flux = magnif * lc.params['Fs'][0] + lc.params['Fb'][0]
            
            # remove points with negative flux and NaN, and compute mag
            date = dat[1]
            errbar = dat[3]
            arg = np.logical_not(np.isnan(flux)) & (flux > 0.)
            flux = flux[arg]
            date = date[arg]
            errbar = errbar[arg]
            mag = - 2.5 * np.log10(flux)
            
            # plot with error bars
            si[i].circle(date, mag, size=4, alpha=1., color=color, legend=tel)
            
            y_err_x, y_err_y = [], []
            for px, py, err in zip(date, mag, errbar):
                y_err_x.append((px, px))
                y_err_y.append((py - err, py + err))
            si[i].multi_line(y_err_x, y_err_y, color=color, alpha=0.4)

            k += 1

    # group plots in one column and generate best-fits html code
    sigrid = [s for s in si]
    p = column(sigrid)
    script, divini = components(p)
    div = '<div align="center">' + divini + '</div>'

    # create search map html code
    wsmap = '<img class="imageStyle" alt="' + event + '" src="./' + event + '.png' + '" width="800" />'
    
    # label version
    version = str(version_info).replace(', ', '.').replace('(', '(version ')
    
    # list of best fit paramters to download
    fitrank = event + '_rank.txt'

    # create model selection list
    selmod = '<select name="display" onchange="location=this.value;">'
    selmod += '<option value="/miiriads/MiSMap/Events/' + event + '.html" selected>Overview</option>'
    for j in range(len(models)):
        selmod += '<option value="/miiriads/MiSMap/Events/' + event + '_' + str(j + 1) + '.html">Model ' + str(j + 1) + '</option>'
    selmod += '</select>'

    # get multiple references of event
    if crossrefs:
        evname = getcrossrefs(crossrefs, event)
    else:
        evname = event

    # fill template html page
    fill_webpage(html_event_template, event + '/' + event + '.html', [('_LASTUPDATE_', fittime), ('_VERSION_', version), ('_EVENT_', evname), ('_SEARCHMAP_', wsmap), ('_MODELS_', div), ('_SELECT_', selmod), ('_RANK_', fitrank), ('<!-- _SCRIPT_ -->', script)])

def makehtml_model(html_model_template, event, models, datasets, trange=None, crossrefs=None):
    """Create individual model of individual event html page"""
    
    # set I/O shell display
    tcol, tun, tend, tit = "\033[0m\033[35m", "\033[0m\033[1m\033[35m", "\033[0m", "\033[0m\033[3m"
    
    # generate one page per individual model
    mc = MagnificationCurve()
    lc = LightCurve(datasets, trange=trange, dmag=None)
    plt_width = 800
    
    for i in range(len(models)):
        
        model = models[i]
        
        # html page name
        printi(tcol + "Creating " + tit + "'" + event + '/' + event + '_' + str(i + 1) + '.html' + "'" + tcol + " web page" + tend)

        # fit light curve using best-fit parameters
        mc.create(model)
        lc.fit(mc, 'central', '+', init=None)
    
        # ligth curve plot template (bind x-axis on light curve plot, with automatic t-range)
        plc = bplt.figure(width=plt_width, plot_height=500, title='Light curve', tools=["pan", "wheel_zoom", "box_zoom", "undo", "redo", "reset", "save"], active_drag="pan", active_scroll="wheel_zoom")
        
        if trange:
            tmin = lc.params['t0'] - 1.5 * lc.params['tE']
            tmax = lc.params['t0'] + 1.5 * lc.params['tE']
            plc.x_range = Range1d(tmin, tmax)

        # residuals plot template
        pres = bplt.figure(width=plt_width, plot_height=200, title='Residuals', x_range=plc.x_range, y_range=(-0.1, 0.1), tools=["pan", "wheel_zoom", "reset", "save"], active_drag="pan", active_scroll="wheel_zoom")

        # caustics plot template
        plt_height = 400
        pcc = bplt.figure(width=plt_width, plot_height=plt_height, title='Source trajectory', tools=["pan", "wheel_zoom", "box_zoom", "undo", "redo", "reset", "save"], active_drag="pan", active_scroll="wheel_zoom", match_aspect=True, x_range=(- float(plt_width)/plt_height, float(plt_width)/plt_height), y_range=(-1, 1))

        # plot caustics
        cc = Caustics(mc.params['s'], mc.params['q'], N=256, cusp=True)
        z = np.ravel(cc.zetac)
        pcc.circle(np.real(z), np.imag(z), size=1, alpha=1., color='red')
   
        # plot trajectory
        traj = lambda t: (-lc.params['u0'] * np.sin(lc.params['alpha']) + np.cos(lc.params['alpha']) * t, lc.params['u0'] * np.cos(lc.params['alpha']) + np.sin(lc.params['alpha']) * t)

        pcc.line(traj(np.array([-10., 10.]))[0], traj(np.array([-10., 10.]))[1], color='firebrick', line_width=1)
   
        pcc.add_layout(Arrow(line_width=1, line_color='firebrick', end=VeeHead(size=10, line_color='firebrick'), x_start=traj((tmin - lc.params['t0'])/lc.params['tE'])[0], y_start=traj((tmin - lc.params['t0'])/lc.params['tE'])[1], x_end=traj((tmax - lc.params['t0'])/lc.params['tE'])[0], y_end=traj((tmax - lc.params['t0'])/lc.params['tE'])[1]))

        # compute blended flux
        flux = lc._mu * lc.params['Fs'][0] + lc.params['Fb'][0]
        t = lc._t * lc.params['tE'] + lc.params['t0']

        # remove points with negative flux and NaN
        arg = np.logical_not(np.isnan(flux)) & (flux > 0.)
        flux = flux[arg]
        t = t[arg]

        # invert light curve y-axis
        plc.y_range = Range1d(1.02 * np.max(-2.5 * np.log10(flux)), 0.98 * np.min(-2.5 * np.log10(flux)))

        # theoretical ligth curve
        plc.line(t, -2.5 * np.log10(flux), color='firebrick')

        # add data on plot
        atel = iter(color_tel)
        for k in range(len(lc._datalist)):
            dat = lc._datalist[k]
            # color and legend
            _ , datai = split(dat[6])
            tel = datai[0]
            if tel not in color_tel:
                tel = atel.next()
            color = color_tel[tel]
 
            # deblending
            magnif = (np.power(10., - dat[2] / 2.5) - lc.params['Fb'][k]) / lc.params['Fs'][k]
            flux = magnif * lc.params['Fs'][0] + lc.params['Fb'][0]
            
            # remove points with negative flux and NaN, and compute mag
            date = dat[1]
            errbar = dat[3]
            arg = np.logical_not(np.isnan(flux)) & (flux > 0.)
            flux = flux[arg]
            date = date[arg]
            errbar = errbar[arg]
            mag = - 2.5 * np.log10(flux)

            mu = lc.content['mc']((date - lc.params['t0']) / lc.params['tE'])
            res = - 2.5 * np.log10(mu * lc.params['Fs'][0] + lc.params['Fb'][0]) - mag

            # plot light curve with error bars
            plc.circle(date, mag, size=4, alpha=1., color=color, legend=tel)

            y_err_x, y_err_y = [], []
            for px, py, err in zip(date, mag, errbar):
                y_err_x.append((px, px))
                y_err_y.append((py - err, py + err))
            plc.multi_line(y_err_x, y_err_y, color=color, alpha=0.4)
           
            # plot residuals with error bars
            pres.line((0., 1e6), (0., 0.), color='black', line_width=0.2)
            
            pres.circle(date, res, size=4, alpha=1., color=color)
            
            y_err_x, y_err_y = [], []
            for px, py, err in zip(date, res, errbar):
                y_err_x.append((px, px))
                y_err_y.append((py - err, py + err))
            pres.multi_line(y_err_x, y_err_y, color=color, alpha=0.4)
            
            # color points at data dates on trajectory
            pcc.circle(traj((date - lc.params['t0'])/lc.params['tE'])[0], traj((date - lc.params['t0'])/lc.params['tE'])[1], size=4, alpha=1., color=color)

        # group plots in one column and generate best-fits html code
        p = column([plc, pres, pcc])
        script, divini = components(p)
        div = '<div align="center">' + divini + '</div>'

        # create search map html code
        wsmap = '<img class="imageStyle" alt="' + event + '" src="./' + event + '.png' + '" width="800" />'

        # label version
        version = str(version_info).replace(', ', '.').replace('(', '(version ')

        # list of best fit paramters to download
        fitrank = event + '_rank.txt'
        
        # date and time of event's update
        if os.path.isfile(event + '/lastupdate.txt'):
            with open(event + '/lastupdate.txt', 'r') as f:
                fittime = f.read()
                f.close()
        else:
            fittime = ''
        
        # get multiple references of event
        if crossrefs:
            evname = getcrossrefs(crossrefs, event)
        else:
            evname = event
    
        # write parameters
        eventdet = 'Microlensing event ' + evname + ' : details of binary-lens model ' + str(i + 1)

        lpar = '<i>Model paramaters</i>'

        lpar += '<p align="left" style="color:#B22222">$s$={0:<.8f}<BR>$q$={1:<.8f}<BR>$u_0$={2:<.8f}<BR>$\\alpha$={3:<.5f}<BR>$t_E$={4:<.3f}<BR>$t_0$={5:<.6f}<BR>$\\rho$={6:<.8f}<BR>'.format(lc.params['s'], lc.params['q'], lc.params['u0'], lc.params['alpha'], lc.params['tE'], lc.params['t0'], lc.params['rho'])
        if 'piEN' in lc.params.keys():
            lpar += '$\\pi_{E, N}$={6:<.8f}<BR>'
        if 'piEE' in lc.params.keys():
            lpar += '$\\pi_{E, E}$={6:<.8f}<BR>'
        if 'ds' in lc.params.keys():
            lpar += '$ds/dt$={6:<.8f}<BR>'
        if 'dalpha' in lc.params.keys():
            lpar += '$d\\alpha/dt$={6:<.8f}<BR>'
        lpar += '</p><BR>'

        lpar += '<i>Light curve, residuals, source trajectory and caustics</i>'
    
        # create model selection list
        selmod = '<select name="display" onchange="location=this.value;">'
        selmod += '<option value="/miiriads/MiSMap/Events/' + event + '.html">Overview</option>'
        for j in range(len(models)):
            if j == i:
                selmod += '<option value="/miiriads/MiSMap/Events/' + event + '_' + str(j + 1) + '.html" selected>Model ' + str(j + 1) + '</option>'
            else:
                selmod += '<option value="/miiriads/MiSMap/Events/' + event + '_' + str(j + 1) + '.html">Model ' + str(j + 1) + '</option>'
        selmod += '</select>'

        # fill template html page
        fill_webpage(html_model_template, event + '/' + event + '_' + str(i + 1) + '.html', [('_LASTUPDATE_', fittime), ('_VERSION_', version), ('_EVENT_', evname), ('_EVENTDET_', eventdet), ('_MODEL_', div), ('_PARAMS_', lpar), ('_SELECT_', selmod), ('_RANK_', fitrank), ('<!-- _SCRIPT_ -->', script)])

def makehtml_index(html_index_template, crossrefs=None):
    """Create index html page
        
        If crossrefs=None, does use names in 'eventslist.dat'
        """
    # set I/O shell display
    tcol, tun, tend, tit = "\033[0m\033[34m", "\033[0m\033[1m\033[34m", "\033[0m", "\033[0m\033[3m"
    
    # add in <head>:
    """<meta http-equiv="refresh" content="30">"""

    # get list of analyzed events
    df = pd.read_csv('eventslist.dat', header=None, skip_blank_lines=True, comment='#', names=['col'])

    # create lists of events (real-time, stopped)
    wlist1, wlist2, wlist3 = '\n', '\n', '\n'
    for j in range(len(df['col'])):
        lev = df['col'].values[j].split(' ')
        
        # last fit date and time
        if os.path.isfile(lev[1] + '/lastupdate.txt'):
            with open(lev[1] + '/lastupdate.txt', 'r') as f:
                fittime = f.read()
                f.close()
        else:
            fittime = ''
        
        # display cross-reference names
        if crossrefs:
            evname = getcrossrefs(crossrefs, lev[1])
        else:
            evname =  lev[1]

        # create lists of event names
        if lev[0] == 'fit_artemis':
            wlist1 = wlist1 + '<a href="../MiSMap/Events/' + lev[1] + '.html" rel="self" title="ARTEMiS alert">' + evname + '</a> ' + fittime + '<br /> \n'
        if lev[0] == 'fit_add':
            wlist2 = wlist2 + '<a href="../MiSMap/Events/' + lev[1] + '.html" rel="self" title="Other events of interest">' + evname + '</a> ' + fittime + '<br /> \n'
        if lev[0] == 'stop':
            wlist3 = wlist3 + '<a href="../MiSMap/Events/' + lev[1] + '.html" rel="self" title="Former events">' + evname + '</a> ' + fittime + '<br /> \n'

    # fill page with content
    fill_webpage(html_index_template, 'index.html', [('_LIST1_', wlist1), ('_LIST2_', wlist2), ('_LIST3_', wlist3)])

def split(pathnfile):
    """Return file and directory path from string"""
    sp = pathnfile.split('/')
    if len(sp) == 1:
        filename = sp[0]
        path = './'
    else:
        filename = sp.pop(len(sp) - 1)
        path = '/'.join(sp) + '/'
    return path, filename

def orderlist(datasets):
    """Permute observatories list until OGLE (in priority) or MOA (second priority)
        are first, otherwise keep current order"""
    for perm in permutations(datasets, len(datasets)):
        if perm[0][0] == 'O':
            return list(perm)
    for perm in permutations(datasets, len(datasets)):
        if perm[0][0] == 'M': # new MOA
            return list(perm)
    return datasets

color_tel = {
    'O': "#000000",
    'M': "#7F0000", # new MOA
#    'K': "#7F0000", # former MOA
#    'L': "#0000FF",
    'L': "#71FEFF",
    'Z': "#FF7F00",
    'H': "#00FFFF",
    'I': "#007F00",
    'J': "#00A0A0",
    'P': "#C0C0C0",
    'Q': "#BF0F00",
    'D': "#FF00FF",
    'E': "#FF00FF",
    'F': "#FF00FF",
    'R': "#FFAF00",
#    'S': "#FFAF00",
    'S': "#00FF00", # new
    'T': "#FFAF00",
    'X': "#7F007F",
    'Y': "#7F007F",
    'U': "#C07F7F",
    'W': "#00007F",
#    'A': "#00FF00",
    'A': "#0000FF", # new
#    'C': "#7F7F00",
    'C': "#FF0000", # new
    'y': "#7F7F00",
    'z': "#007070",
    'l': "#B0FFB0",
    'm': "#00FFFF",
    'o': "#FF7F00",
    'r': "#7F7FC0",
    'd': "#FFAF00",
    'a': "#007F00",
    'h': "#C07F7F",
    't': "#C0C0C0",
    'f': "#0000FF",
    'k': "#00007F",
    'v': "#7F007F",
    'p': "#FF00FF",
    'w': "#BF0F00",
    'i': "#BF0F00",
    'b': "#00A0A0",
    's': "#00FF00",
}

# ========= for development only ============
if __name__ == '__main__':
    
    verbosity('DEBUG')

