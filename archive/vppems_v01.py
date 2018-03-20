import sys
from osbrain import run_agent
from osbrain import run_nameserver

from archive.EMSParties import VPPClassInt
from archive.EMSParties import VPPClassExt
from archive.EMSParties import DSOClass
        

def rep_for_req():
    print('Reply for request!')

                   

if __name__ == '__main__':

    # System deployment
    ns = run_nameserver()
    global_data = {'VPP': [('VPP1'), ('VPP2'), ('VPP3')], 'DSO': True,
                   'Ext': [('VPP1_Ext'), ('VPP2_Ext'), ('VPP3_Ext')]}

    # Running agents
    for str0 in global_data['VPP']:
        str1 = str0.lower()+'int'+"=run_agent('"+str0+'_Int'+"'"+', base=VPPClassInt'")"
        str2 = str0.lower()+'ext'+"=run_agent('"+str0+'_Ext'+"'"+', base=VPPClassExt'")"
        exec(str1)
        exec(str2)
    
    #vpp1int = run_agent('VPP1_Int', base=VPPClassInt) Substituted by loop above
    #vpp1ext = run_agent('VPP1_Ext', base=VPPClassExt)
    
    if global_data['DSO']==True:
        dso = run_agent('DSO', base=DSOClass)

    print('Registered agents: ')
    for a in ns.agents():
        print(a)
    print('-------------------')

    ## Communication

    
    # 1) ext <-> int (single side, by PUSH/PULL)
    method1 = getattr(VPPClassExt, 'pullFromInt')
    method2 = getattr(VPPClassInt, 'pullFromExt')
    for str0 in global_data['VPP']:
        str1 = str0.lower()+'ext.connect('+str0.lower()+'int.addr("intPushToExt"),handler=method1)'
        exec(str1)
        str2 = str0.lower() + 'int.connect(' + str0.lower() + 'ext.addr("extPushToInt"),handler=method2)'
        exec(str2)
    #vpp1ext.connect(vpp1int.addr('intPushToExt'), handler=pull_from_int) # connect requests int->ext by push
    #vpp1int.connect(vpp1ext.addr('extPushToInt'), handler=pull_from_ext) # connect push ext->int



    # 2) ext <-> ext (Request-reply to other VPPs, sensitivity included, so not everybody cares)

    for str0 in global_data['VPP']:
        for str00 in global_data['VPP']:
            if str0==str00:
               continue
            str1 = str0.lower() + 'ext.connect(' + str00.lower() + 'ext.addr("repIfRequest"),alias="' + str0 + "to" + str00 + '")'
            exec(str1)
        str1 = str0.lower() + 'ext.connect(dso.addr("repIfRequest"),alias="' + str0 + 'toDSO")'
        exec(str1)

    #vpp1ext.connect(vpp2ext.addr('repIfRequest'), alias='1to2') # loop above instead
    #vpp1ext.connect(vpp3ext.addr('repIfRequest'), alias='1to3')
    #vpp1ext.connect(dso.addr('repIfRequest'), alias='1toD')

    ### Periodical operation of the internal agents (?) As the ones doing things constantly
    
    method = getattr(VPPClassInt, 'periodical') # calling periodical function

    start_timestamp = 1 # starting from first timestamp
    period = 0.25 # time period of executing .each
    step = 1 # incrementation step from data
    
    vpp1int.each(period, method, *(start_timestamp, step))
    #vpp2int.each(period, method, *(start_timestamp, step))
    #vpp3int.each(period, method, *(start_timestamp, step))

    sys.exit(0)

    requests=(0.0,0.5,0.6,0.9)

    vpp1ext.send('1to2', requests[1])
    vpp1ext.send('1to3', requests[2])
    vpp1ext.send('1toD', requests[3])

    rep_from2 = vpp1ext.recv('1to2')
    rep_from3 = vpp1ext.recv('1to3')
    rep_fromD = vpp1ext.recv('1toD')

    print(rep_from2)
    print(rep_from3)
    print(rep_fromD)