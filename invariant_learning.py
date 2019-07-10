#
# Unsupervised learning of olfactory inference
#
# with learning of concentration-invariant representation
#
# See the associated README file (http://github.com/nhiratani/olfactory_learning) for the details. 
#
from math import *
import os
os.environ["MKL_NUM_THREADS"] = '4'
os.environ["NUMEXPR_NUM_THREADS"] = '4'
os.environ["OMP_NUM_THREADS"] = '4'
import sys
import numpy as np
from numpy import random as nrnd
from numpy import linalg as nlg
from scipy import special as scisp

T = 8000 #total number of trial

class GenerativeModel():
    def __init__(model, co, M, N, sigmax):
        model.alphac = 3.0 #uniformity of odor concentration
        model.co = co; model.M = M; model.N = N; model.sigmax2 = sigmax*sigmax
        model.w = model.generate_w()

    def generate_w(model): 
        co = model.co; M = model.M; N = model.N
        w = np.divide( nrnd.lognormal(0.0,1.0,(N,M)), co*M )
        Zw = np.dot(w, np.ones(M))
        wtot = np.average( Zw )
        w = wtot*(np.transpose( np.divide(np.transpose(w),Zw) ) )
        return w

    def generate_scxt(model): 
        co = model.co; M = model.M; N = model.N; sigmax = sqrt(model.sigmax2)
        st = nrnd.binomial(1,co,(M))
        while( np.sum(st) < 0.5 ): #skip a trial where none of odors is present
            st = nrnd.binomial(1,co,(M))
        ct = np.multiply( st, nrnd.gamma(model.alphac, 1.0/model.alphac,(M)) )
        xt = nrnd.normal( np.dot(model.w, ct), np.full((N), sigmax) )
        return st, ct, xt

class InferenceModel():
    def __init__(model, co, M, N, sigmax, Zrho_init, circuit_type): 
        model.g_sig_init = 0.1 #initial log-variance parameter of wF and wL
        model.Zdelta_t_min = 100.0; #minimum(initial) value of delta_t [trials]
        model.dTau = 1.0 #time step [ms]
        model.Taus = np.arange(0.0,5001.0,model.dTau) #simulation time at each trial
        model.taur = 50.0 # timescale of firing rate dynamics [ms]
        model.mtsp = 5.0
        
        Zwinit = exp( 0.5*(model.g_sig_init*model.g_sig_init - 1.0) )
        model.delta_t = 1.0/model.Zdelta_t_min

        model.co = co; model.M = M; model.N = N; model.sigmax2 = sigmax*sigmax;
        model.wF = np.divide( nrnd.lognormal(0.0,model.g_sig_init,(M,N)), co*M*Zwinit )
        model.wL = np.divide( nrnd.lognormal(0.0,model.g_sig_init,(N,M)), co*M*Zwinit )
        model.wp = np.divide( nrnd.lognormal(0.0,model.g_sig_init,(M,N)), co*M*Zwinit )
        model.rhoc = np.full((M), (co/model.sigmax2)/Zrho_init )
        model.rhop = np.full((M), (co/model.sigmax2)/Zrho_init )
        
        model.lambda_c = np.zeros((M)); model.lambda_p = np.zeros((M)); model.update_lambda_cp()

        model.ctype = circuit_type
        model.Jp = np.full((M,M), 0.02) - np.diag(np.full((M), 0.02))

    def kai(model, altmp): #approximated 1/Psi
        M = model.M
        kaitmp = np.where(altmp/sqrt(2.0) < -10.0, -altmp, np.zeros((M)))
        kaitmp2 = np.where(abs(altmp/sqrt(2.0)) < 10.0, sqrt(2.0/pi)/scisp.erfcx(-altmp/sqrt(2.0)), kaitmp)
        return kaitmp2

    def f0(model, altmp, ptmp, ltmp): #Prob[c>0]
        co = model.co; M = model.M
        
        al2tmp = np.multiply(altmp, altmp)
        kaitmp = model.kai(altmp);
        l3d2tmp = np.multiply(ltmp, np.sqrt(ltmp))
        
        numers = np.multiply(np.multiply(altmp, kaitmp) + np.ones((M)) + al2tmp, ptmp)
        denom1 = (2.0/27.0)*np.multiply(np.ones((M)) - ptmp, np.multiply(l3d2tmp, kaitmp))
        denom2 = np.multiply( np.multiply(altmp, kaitmp) + (np.ones((M)) + al2tmp), ptmp )
        
        return np.divide(numers, denom1 + denom2)

    def f1(model, altmp, ptmp, ltmp): #<c>
        co = model.co; M = model.M

        al2tmp = np.multiply(altmp, altmp)
        kaitmp = model.kai(altmp);
        l3d2tmp = np.multiply(ltmp, np.sqrt(ltmp))
        
        numer1 = np.multiply( 2.0*np.ones((M)) + al2tmp, kaitmp )
        numer2 = np.multiply( 3.0*np.ones((M)) + al2tmp, altmp )
        numer = np.multiply(numer1 + numer2, ptmp)

        denom1 = (2.0/27.0)*np.multiply(np.ones((M)) - ptmp, np.multiply(l3d2tmp, kaitmp))
        denom2 = np.multiply( np.multiply(altmp, kaitmp) + (np.ones((M)) + al2tmp), ptmp )
        denom = np.multiply(denom1+denom2, np.sqrt(ltmp))
        return np.divide( numer, denom )

    def f2(model, altmp, ptmp, ltmp): #<c*c>
        co = model.co; M = model.M
        f0tmp = model.f0(altmp, ptmp, ltmp)
        f1tmp = model.f1(altmp, ptmp, ltmp)
        f2atmp = np.divide( np.multiply(altmp, f1tmp), np.sqrt(ltmp) )
        f2btmp = 3.0*np.divide(f0tmp, ltmp)
        return f2atmp + f2btmp

    def fmcp(model, mcpt, xt): 
        co = model.co; M = model.M; N = model.N; sigmax2 = model.sigmax2; taur = model.taur
        mt = mcpt[:N]; ctbar = mcpt[N:N+M]; ptbar = mcpt[N+M:N+2*M]
        dmt = (1.0/taur)*(-mt + xt - np.dot(model.wL, ctbar))
        wF2sum = np.diag( np.dot(model.wF, np.transpose(model.wF)) ) 
        altmp = np.divide( np.dot(model.wF, mt) + np.multiply(wF2sum, ctbar) - np.full((M), 3.0*sigmax2),\
                           sigmax2*np.sqrt( model.lambda_c ) )
        if model.ctype == 0: #without p to c
            dctbar = (1.0/taur)*(-ctbar + model.f1(altmp, np.full((M), co), model.lambda_c))
        else: #with p to c
            dctbar = (1.0/taur)*(-ctbar + model.f1(altmp, ptbar, model.lambda_c))

        wp2sum = np.diag( np.dot(model.wp, np.transpose(model.wp)) ) 
        alptmp = np.divide( np.dot(model.wp, mt) + np.multiply(wp2sum, ptbar) - np.full((M), 3.0*sigmax2)\
                            - sigmax2*np.multiply(model.lambda_p, np.dot(model.Jp, ptbar)), sigmax2*np.sqrt( model.lambda_p ))
        dptbar = (1.0/taur)*(-ptbar + model.f0(alptmp, np.full((M), co), model.lambda_p))
        return np.concatenate((dmt, dctbar, dptbar))

    def calc_mcpt(model, xt): 
        co = model.co; M = model.M; N = model.N; dTau = model.dTau
        mcpt = np.concatenate( (np.zeros((N)),  np.full((2*M), co)) )
        for tau in model.Taus:
            k1 = model.fmcp(mcpt, xt)
            k2 = model.fmcp(mcpt + 0.5*dTau*k1, xt)
            k3 = model.fmcp(mcpt + 0.5*dTau*k2, xt)
            k4 = model.fmcp(mcpt + 1.0*dTau*k3, xt)
            mcpt = mcpt + (dTau/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)
            mcpt[:N] = np.clip( mcpt[:N], -model.mtsp, None )
            mcpt[N:] = np.clip( mcpt[N:], 0.0, None )
        return mcpt

    def update_wFL_rhoc(model, mcpt, xt): 
        co = model.co; M = model.M; N = model.N; sigmax2 = model.sigmax2; delta_t = model.delta_t;
        mt = mcpt[:N]; ctbar = mcpt[N:N+M]; ptbar = mcpt[N+M:]

        rhoc_old = model.rhoc
        wF2sum = np.diag( np.dot(model.wF, np.transpose(model.wF)) ) 
        altmp = np.divide( np.dot(model.wF, mt) + np.multiply(wF2sum, ctbar) - np.full((M), 3.0*sigmax2),\
                           sigmax2*np.sqrt( model.lambda_c ) )
        if model.ctype == 0: #without p to c
            model.rhoc = (1.0-delta_t)*rhoc_old + (delta_t/sigmax2)*model.f2(altmp, np.full((M), co), model.lambda_c)
        else: #with p to c
            model.rhoc = (1.0-delta_t)*rhoc_old + (delta_t/sigmax2)*model.f2(altmp, ptbar, model.lambda_c)
        
        Fdisc = (1.0 - delta_t)*np.divide(rhoc_old, model.rhoc)\
                + (delta_t/sigmax2)*np.divide( np.multiply(ctbar, ctbar), model.rhoc )

        wFold = np.multiply( np.outer(Fdisc, np.ones((N))), model.wF )
        dwF = (delta_t/sigmax2)*np.outer( np.divide(ctbar, model.rhoc), mt )
        model.wF = np.clip( wFold + dwF, 0.0, None )

        wLold = np.multiply( np.outer(np.ones((N)), Fdisc), model.wL )
        dwL = (delta_t/sigmax2)*np.outer( mt, np.divide(ctbar, model.rhoc) )
        model.wL = np.clip( wLold + dwL, 0.0, None )

    def update_wp_rhop(model, mcpt, xt): 
        co = model.co; M = model.M; N = model.N; sigmax2 = model.sigmax2; delta_t = model.delta_t;
        mt = mcpt[:N]; ptbar = mcpt[N+M:]

        rhop_old = model.rhop
        wp2sum = np.diag( np.dot(model.wp, np.transpose(model.wp)) ) 
        alptmp = np.divide( np.dot(model.wp, mt) + np.multiply(wp2sum, ptbar) - np.full((M), 3.0*sigmax2)\
                            - sigmax2*np.multiply(model.lambda_p, np.dot(model.Jp, ptbar)), sigmax2*np.sqrt( model.lambda_p ))
        model.rhop = (1.0-delta_t)*rhop_old + (delta_t/sigmax2)*model.f2(alptmp, np.full((M), co), model.lambda_p)

        est_ct = model.f1(alptmp, np.full((M), co), model.lambda_p)
        Fdisc = (1.0 - delta_t)*np.divide(rhop_old, model.rhop)\
                + (delta_t/sigmax2)*np.divide( np.multiply(est_ct, est_ct), model.rhop )

        wpold = np.multiply( np.outer(Fdisc, np.ones((N))), model.wp )
        dwp = (delta_t/sigmax2)*np.outer( np.divide(est_ct, model.rhop), mt )
        model.wp = np.clip( wpold + dwp, 0.0, None )

        model.Jp += 0.1*(np.outer(ptbar, ptbar) - 5.0*co*np.multiply(model.Jp, ptbar))
        model.Jp = np.clip(model.Jp - np.diag( np.diag(model.Jp) ), 0.0, None)

    def update_lambda_cp(model): 
        co = model.co; M = model.M; N = model.N; sigmax2 = model.sigmax2; delta_t = model.delta_t
        model.lambda_c = (np.diag( np.dot(model.wF, np.transpose(model.wF)) )\
                          + np.divide(np.full((M), N*delta_t), model.rhoc))/sigmax2
        model.lambda_p = (np.diag( np.dot(model.wp, np.transpose(model.wp)) )\
                          + np.divide(np.full((M), N*delta_t), model.rhop))/sigmax2

class PerformanceMeasure():
    def __init__(pm, M, N): 
        pm.M = M; pm.N = N
        pm.wFprefq = np.arange(0,M,1); nrnd.shuffle( pm.wFprefq )
        pm.wpprefq = np.arange(0,M,1); nrnd.shuffle( pm.wpprefq )

    def calc_gprefq(pm, wL, W, wFtof): #checked
        M = pm.M; N = pm.N; 
        mwL = np.outer(np.ones(N), np.dot( np.ones(N), wL))/float(N)
        mW = np.outer(np.ones(N), np.dot( np.ones(N), W))/float(N)
        Cov_gw = np.dot(np.transpose(wL-mwL), W-mW)
        for j1 in range(M):
            prefidx = 0; covmaxtmp = -100.0
            for j2 in range(M):
                if Cov_gw[j1][j2] > covmaxtmp:
                    covmaxtmp = Cov_gw[j1][j2]; prefidx = j2
            if wFtof:
                pm.wFprefq[j1] = prefidx
            else:
                pm.wpprefq[j1] = prefidx

    def calc_ct_est(pm, zt, prefq): #checked
        M = pm.M
        ct_est = np.zeros((M))
        for j1 in range(M):
            qcnt = 0
            for j2 in range(M):
                if prefq[j2] == j1:
                    ct_est[j1] += zt[j2]; qcnt += 1
            if qcnt != 0:
                ct_est[j1] = ct_est[j1]/float(qcnt)
        return ct_est
    
    def calc_err(pm, st, ct, zt, prefq): #checked
        M = pm.M;
        errs = [0,0,0,0]
        ct_est = pm.calc_ct_est(zt, prefq)
        corrtmp = np.corrcoef(ct, ct_est); errs[0] = corrtmp[0][1]
        errs[1] = np.dot(ct-ct_est, ct-ct_est)/float(M)

        corrtmp = np.corrcoef(st, ct_est); errs[2] = corrtmp[0][1]
        errs[3] = np.dot(st-ct_est, st-ct_est)/float(M)
        return errs
    
    def calc_werr(pm, wL, W, prefq): #checked
        M = pm.M; N = pm.N; werr = 0.0
        for j in range(M):
            ZwLtmp = 0.0; Zwtmp = 0.0; werrtmp = 0.0; jidx = prefq[j]
            for i in range(N):
                ZwLtmp += wL[i][j]; Zwtmp += W[i][jidx]
            Zrtmp = ZwLtmp/Zwtmp
            for i in range(N):
                werrtmp += (wL[i][j]/Zrtmp - W[i][jidx])*(wL[i][j]/Zrtmp - W[i][jidx])
            werr += sqrt( werrtmp/float(N) )
        return werr/float(M)
    
def simul(coM, M, N, sigmax, Zrho_init, circuit_type, ik):
    festr = 'data/invariant_learning_readout_coM' + str(coM) + '_M' + str(M) + '_N' + str(N) + '_sx' + str(sigmax)\
            + '_zr' + str(Zrho_init) + '_ct' + str(circuit_type) + '_ik' + str(ik) + '.txt'
    fwe = open(festr,'w')

    co = coM/float(M)
    gmodel = GenerativeModel(co, M, N, sigmax)
    imodel = InferenceModel(co, M, N, sigmax, Zrho_init, circuit_type)
    perfm = PerformanceMeasure(M,N)
    
    for t in range(T):
        imodel.delta_t = 1.0/(t + imodel.Zdelta_t_min)
        true_st, ct, xt = gmodel.generate_scxt()
        mcpt = imodel.calc_mcpt(xt)
        cferr = perfm.calc_err(true_st, ct, mcpt[N:N+M], perfm.wFprefq)
        cperr = perfm.calc_err(true_st, ct, mcpt[N+M:], perfm.wpprefq)
        fwetmp = str(t) + " " + str(cferr[0]) + " " + str(cferr[1]) + " " + str(cferr[2]) + " " + str(cferr[3])\
                 + " " + str(cperr[0]) + " " + str(cperr[1]) + " " + str(cperr[2]) + " " + str(cperr[3]) \
                 + " " + str(perfm.calc_werr(np.transpose(imodel.wF), gmodel.w, perfm.wFprefq))\
                 + " " + str(perfm.calc_werr(np.transpose(imodel.wp), gmodel.w, perfm.wpprefq)) + "\n"
        fwe.write( fwetmp );
        
        imodel.update_wFL_rhoc(mcpt, xt)
        imodel.update_wp_rhop(mcpt, xt)
        imodel.update_lambda_cp()
        perfm.calc_gprefq(np.transpose(imodel.wF), gmodel.w, True)
        perfm.calc_gprefq(np.transpose(imodel.wp), gmodel.w, False)
        
        if t%10 == 0 or t==T-1:
            fwe.flush();
    
def main():
    param = sys.argv
    coM = int(param[1])
    M = int(param[2])
    N = int(param[3])
    sigmax = float(param[4])
    Zrho_init = float(param[5])
    circuit_type = int(param[6])
    ik = int(param[7])

    simul(coM, M, N, sigmax, Zrho_init, circuit_type, ik)

if __name__ == "__main__":
    main()
