/*************************************************************************
ALGLIB 3.15.0 (source code generated 2019-02-20)
Copyright (c) Sergey Bochkanov (ALGLIB project).

>>> SOURCE LICENSE >>>
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation (www.fsf.org); either version 2 of the 
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

A copy of the GNU General Public License is available at
http://www.fsf.org/licensing/licenses
>>> END OF LICENSE >>>
*************************************************************************/
#ifndef _optimization_pkg_h
#define _optimization_pkg_h
#include "ap.h"
#include "alglibinternal.h"
#include "linalg.h"
#include "alglibmisc.h"
#include "solvers.h"

/////////////////////////////////////////////////////////////////////////
//
// THIS SECTION CONTAINS COMPUTATIONAL CORE DECLARATIONS (DATATYPES)
//
/////////////////////////////////////////////////////////////////////////
namespace alglib_impl
{
#if defined(AE_COMPILE_CQMODELS) || !defined(AE_PARTIAL_BUILD)
typedef struct
{
    ae_int_t n;
    ae_int_t k;
    double alpha;
    double tau;
    double theta;
    ae_matrix a;
    ae_matrix q;
    ae_vector b;
    ae_vector r;
    ae_vector xc;
    ae_vector d;
    ae_vector activeset;
    ae_matrix tq2dense;
    ae_matrix tk2;
    ae_vector tq2diag;
    ae_vector tq1;
    ae_vector tk1;
    double tq0;
    double tk0;
    ae_vector txc;
    ae_vector tb;
    ae_int_t nfree;
    ae_int_t ecakind;
    ae_matrix ecadense;
    ae_matrix eq;
    ae_matrix eccm;
    ae_vector ecadiag;
    ae_vector eb;
    double ec;
    ae_vector tmp0;
    ae_vector tmp1;
    ae_vector tmpg;
    ae_matrix tmp2;
    ae_bool ismaintermchanged;
    ae_bool issecondarytermchanged;
    ae_bool islineartermchanged;
    ae_bool isactivesetchanged;
} convexquadraticmodel;
#endif
#if defined(AE_COMPILE_OPTGUARDAPI) || !defined(AE_PARTIAL_BUILD)
typedef struct
{
    ae_bool nonc0suspected;
    ae_bool nonc0test0positive;
    ae_int_t nonc0fidx;
    double nonc0lipschitzc;
    ae_bool nonc1suspected;
    ae_bool nonc1test0positive;
    ae_bool nonc1test1positive;
    ae_int_t nonc1fidx;
    double nonc1lipschitzc;
    ae_bool badgradsuspected;
    ae_int_t badgradfidx;
    ae_int_t badgradvidx;
    ae_vector badgradxbase;
    ae_matrix badgraduser;
    ae_matrix badgradnum;
} optguardreport;
typedef struct
{
    ae_bool positive;
    ae_int_t fidx;
    ae_vector x0;
    ae_vector d;
    ae_int_t n;
    ae_vector stp;
    ae_vector f;
    ae_int_t cnt;
    ae_int_t stpidxa;
    ae_int_t stpidxb;
} optguardnonc1test0report;
typedef struct
{
    ae_bool positive;
    ae_int_t fidx;
    ae_int_t vidx;
    ae_vector x0;
    ae_vector d;
    ae_int_t n;
    ae_vector stp;
    ae_vector g;
    ae_int_t cnt;
    ae_int_t stpidxa;
    ae_int_t stpidxb;
} optguardnonc1test1report;
#endif
#if defined(AE_COMPILE_OPTSERV) || !defined(AE_PARTIAL_BUILD)
typedef struct
{
    ae_vector norms;
    ae_vector alpha;
    ae_vector rho;
    ae_matrix yk;
    ae_vector idx;
    ae_vector bufa;
    ae_vector bufb;
} precbuflbfgs;
typedef struct
{
    ae_int_t n;
    ae_int_t k;
    ae_vector d;
    ae_matrix v;
    ae_vector bufc;
    ae_matrix bufz;
    ae_matrix bufw;
    ae_vector tmp;
} precbuflowrank;
typedef struct
{
    ae_int_t n;
    ae_int_t k;
    ae_bool checksmoothness;
    ae_vector dcur;
    ae_int_t enqueuedcnt;
    ae_vector enqueuedstp;
    ae_vector enqueuedx;
    ae_vector enqueuedfunc;
    ae_matrix enqueuedjac;
    ae_vector sortedstp;
    ae_vector sortedidx;
    ae_int_t sortedcnt;
    ae_bool linesearchspoiled;
    ae_bool linesearchstarted;
    double nonc0currentrating;
    double nonc1currentrating;
    ae_bool badgradhasxj;
    optguardreport rep;
    double nonc1test0strrating;
    double nonc1test0lngrating;
    optguardnonc1test0report nonc1test0strrep;
    optguardnonc1test0report nonc1test0lngrep;
    double nonc1test1strrating;
    double nonc1test1lngrating;
    optguardnonc1test1report nonc1test1strrep;
    optguardnonc1test1report nonc1test1lngrep;
    ae_bool needfij;
    ae_vector x;
    ae_vector fi;
    ae_matrix j;
    rcommstate rstateg0;
    ae_vector xbase;
    ae_vector fbase;
    ae_vector fm;
    ae_vector fc;
    ae_vector fp;
    ae_vector jm;
    ae_vector jc;
    ae_vector jp;
    ae_matrix jbaseusr;
    ae_matrix jbasenum;
    ae_vector stp;
    ae_vector bufr;
    ae_vector f;
    ae_vector g;
    ae_vector deltax;
    ae_vector tmpidx;
    ae_vector bufi;
    ae_vector xu;
    ae_vector du;
    ae_vector f0;
    ae_matrix j0;
} smoothnessmonitor;
#endif
#if defined(AE_COMPILE_SNNLS) || !defined(AE_PARTIAL_BUILD)
typedef struct
{
    ae_int_t ns;
    ae_int_t nd;
    ae_int_t nr;
    ae_matrix densea;
    ae_vector b;
    ae_vector nnc;
    double debugflops;
    ae_int_t debugmaxinnerits;
    ae_vector xn;
    ae_vector xp;
    ae_matrix tmpca;
    ae_matrix tmplq;
    ae_matrix trda;
    ae_vector trdd;
    ae_vector crb;
    ae_vector g;
    ae_vector d;
    ae_vector dx;
    ae_vector diagaa;
    ae_vector cb;
    ae_vector cx;
    ae_vector cborg;
    ae_vector tmpcholesky;
    ae_vector r;
    ae_vector regdiag;
    ae_vector tmp0;
    ae_vector tmp1;
    ae_vector tmp2;
    ae_vector rdtmprowmap;
} snnlssolver;
#endif
#if defined(AE_COMPILE_SACTIVESETS) || !defined(AE_PARTIAL_BUILD)
typedef struct
{
    ae_int_t n;
    ae_int_t algostate;
    ae_vector xc;
    ae_bool hasxc;
    ae_vector s;
    ae_vector h;
    ae_vector cstatus;
    ae_bool basisisready;
    ae_matrix sdensebatch;
    ae_matrix pdensebatch;
    ae_matrix idensebatch;
    ae_int_t densebatchsize;
    ae_vector sparsebatch;
    ae_int_t sparsebatchsize;
    ae_int_t basisage;
    ae_bool feasinitpt;
    ae_bool constraintschanged;
    ae_vector hasbndl;
    ae_vector hasbndu;
    ae_vector bndl;
    ae_vector bndu;
    ae_matrix cleic;
    ae_int_t nec;
    ae_int_t nic;
    ae_vector mtnew;
    ae_vector mtx;
    ae_vector mtas;
    ae_vector cdtmp;
    ae_vector corrtmp;
    ae_vector unitdiagonal;
    snnlssolver solver;
    ae_vector scntmp;
    ae_vector tmp0;
    ae_vector tmpfeas;
    ae_matrix tmpm0;
    ae_vector rctmps;
    ae_vector rctmpg;
    ae_vector rctmprightpart;
    ae_matrix rctmpdense0;
    ae_matrix rctmpdense1;
    ae_vector rctmpisequality;
    ae_vector rctmpconstraintidx;
    ae_vector rctmplambdas;
    ae_matrix tmpbasis;
    ae_vector tmpnormestimates;
    ae_vector tmpreciph;
    ae_vector tmpprodp;
    ae_vector tmpprods;
    ae_vector tmpcp;
    ae_vector tmpcs;
    ae_vector tmpci;
} sactiveset;
#endif
#if defined(AE_COMPILE_QQPSOLVER) || !defined(AE_PARTIAL_BUILD)
typedef struct
{
    double epsg;
    double epsf;
    double epsx;
    ae_int_t maxouterits;
    ae_bool cgphase;
    ae_bool cnphase;
    ae_int_t cgminits;
    ae_int_t cgmaxits;
    ae_int_t cnmaxupdates;
    ae_int_t sparsesolver;
} qqpsettings;
typedef struct
{
    ae_int_t n;
    ae_int_t akind;
    ae_matrix densea;
    sparsematrix sparsea;
    ae_bool sparseupper;
    double absamax;
    double absasum;
    double absasum2;
    ae_vector b;
    ae_vector bndl;
    ae_vector bndu;
    ae_vector havebndl;
    ae_vector havebndu;
    ae_vector xs;
    ae_vector xf;
    ae_vector gc;
    ae_vector xp;
    ae_vector dc;
    ae_vector dp;
    ae_vector cgc;
    ae_vector cgp;
    sactiveset sas;
    ae_vector activated;
    ae_int_t nfree;
    ae_int_t cnmodelage;
    ae_matrix densez;
    sparsematrix sparsecca;
    ae_vector yidx;
    ae_vector regdiag;
    ae_vector regx0;
    ae_vector tmpcn;
    ae_vector tmpcni;
    ae_vector tmpcnb;
    ae_vector tmp0;
    ae_vector tmp1;
    ae_vector stpbuf;
    sparsebuffers sbuf;
    ae_int_t repinneriterationscount;
    ae_int_t repouteriterationscount;
    ae_int_t repncholesky;
    ae_int_t repncupdates;
} qqpbuffers;
#endif
#if defined(AE_COMPILE_MINLBFGS) || !defined(AE_PARTIAL_BUILD)
typedef struct
{
    ae_int_t n;
    ae_int_t m;
    double epsg;
    double epsf;
    double epsx;
    ae_int_t maxits;
    ae_bool xrep;
    double stpmax;
    ae_vector s;
    double diffstep;
    ae_int_t nfev;
    ae_int_t mcstage;
    ae_int_t k;
    ae_int_t q;
    ae_int_t p;
    ae_vector rho;
    ae_matrix yk;
    ae_matrix sk;
    ae_vector xp;
    ae_vector theta;
    ae_vector d;
    double stp;
    ae_vector work;
    double fold;
    double trimthreshold;
    ae_vector xbase;
    ae_int_t prectype;
    double gammak;
    ae_matrix denseh;
    ae_vector diagh;
    ae_vector precc;
    ae_vector precd;
    ae_matrix precw;
    ae_int_t preck;
    precbuflbfgs precbuf;
    precbuflowrank lowrankbuf;
    double fbase;
    double fm2;
    double fm1;
    double fp1;
    double fp2;
    ae_vector autobuf;
    ae_vector invs;
    ae_vector x;
    double f;
    ae_vector g;
    ae_bool needf;
    ae_bool needfg;
    ae_bool xupdated;
    ae_bool userterminationneeded;
    double teststep;
    rcommstate rstate;
    ae_int_t repiterationscount;
    ae_int_t repnfev;
    ae_int_t repterminationtype;
    linminstate lstate;
    ae_int_t smoothnessguardlevel;
    smoothnessmonitor smonitor;
    ae_vector lastscaleused;
} minlbfgsstate;
typedef struct
{
    ae_int_t iterationscount;
    ae_int_t nfev;
    ae_int_t terminationtype;
} minlbfgsreport;
#endif
#if defined(AE_COMPILE_QPDENSEAULSOLVER) || !defined(AE_PARTIAL_BUILD)
typedef struct
{
    double epsx;
    ae_int_t outerits;
    double rho;
} qpdenseaulsettings;
typedef struct
{
    ae_vector nulc;
    ae_matrix sclsfta;
    ae_vector sclsftb;
    ae_vector sclsfthasbndl;
    ae_vector sclsfthasbndu;
    ae_vector sclsftbndl;
    ae_vector sclsftbndu;
    ae_vector sclsftxc;
    ae_matrix sclsftcleic;
    ae_matrix exa;
    ae_vector exb;
    ae_vector exxc;
    ae_vector exbndl;
    ae_vector exbndu;
    ae_vector exscale;
    ae_vector exxorigin;
    qqpsettings qqpsettingsuser;
    qqpbuffers qqpbuf;
    ae_vector nulcest;
    ae_vector tmp0;
    ae_matrix tmp2;
    ae_vector modelg;
    ae_vector d;
    ae_vector deltax;
    convexquadraticmodel dummycqm;
    sparsematrix dummysparse;
    ae_matrix qrkkt;
    ae_vector qrrightpart;
    ae_vector qrtau;
    ae_vector qrsv0;
    ae_vector qrsvx1;
    ae_vector nicerr;
    ae_vector nicnact;
    ae_int_t repinneriterationscount;
    ae_int_t repouteriterationscount;
    ae_int_t repncholesky;
    ae_int_t repnwrkchanges;
    ae_int_t repnwrk0;
    ae_int_t repnwrk1;
    ae_int_t repnwrkf;
    ae_int_t repnmv;
} qpdenseaulbuffers;
#endif
#if defined(AE_COMPILE_MINBLEIC) || !defined(AE_PARTIAL_BUILD)
typedef struct
{
    ae_int_t nmain;
    ae_int_t nslack;
    double epsg;
    double epsf;
    double epsx;
    ae_int_t maxits;
    ae_bool xrep;
    ae_bool drep;
    double stpmax;
    double diffstep;
    sactiveset sas;
    ae_vector s;
    ae_int_t prectype;
    ae_vector diagh;
    ae_vector x;
    double f;
    ae_vector g;
    ae_bool needf;
    ae_bool needfg;
    ae_bool xupdated;
    ae_bool lsstart;
    ae_bool steepestdescentstep;
    ae_bool boundedstep;
    ae_bool userterminationneeded;
    rcommstate rstate;
    ae_vector ugc;
    ae_vector cgc;
    ae_vector xn;
    ae_vector ugn;
    ae_vector cgn;
    ae_vector xp;
    double fc;
    double fn;
    double fp;
    ae_vector d;
    ae_matrix cleic;
    ae_int_t nec;
    ae_int_t nic;
    double lastgoodstep;
    double lastscaledgoodstep;
    double maxscaledgrad;
    ae_vector hasbndl;
    ae_vector hasbndu;
    ae_vector bndl;
    ae_vector bndu;
    ae_int_t repinneriterationscount;
    ae_int_t repouteriterationscount;
    ae_int_t repnfev;
    ae_int_t repvaridx;
    ae_int_t repterminationtype;
    double repdebugeqerr;
    double repdebugfs;
    double repdebugff;
    double repdebugdx;
    ae_int_t repdebugfeasqpits;
    ae_int_t repdebugfeasgpaits;
    ae_vector xstart;
    snnlssolver solver;
    double fbase;
    double fm2;
    double fm1;
    double fp1;
    double fp2;
    double xm1;
    double xp1;
    double gm1;
    double gp1;
    ae_int_t cidx;
    double cval;
    ae_vector tmpprec;
    ae_vector tmp0;
    ae_int_t nfev;
    ae_int_t mcstage;
    double stp;
    double curstpmax;
    double activationstep;
    ae_vector work;
    linminstate lstate;
    double trimthreshold;
    ae_int_t nonmonotoniccnt;
    ae_matrix bufyk;
    ae_matrix bufsk;
    ae_vector bufrho;
    ae_vector buftheta;
    ae_int_t bufsize;
    double teststep;
    ae_int_t smoothnessguardlevel;
    smoothnessmonitor smonitor;
    ae_vector lastscaleused;
    ae_vector invs;
} minbleicstate;
typedef struct
{
    ae_int_t iterationscount;
    ae_int_t nfev;
    ae_int_t varidx;
    ae_int_t terminationtype;
    double debugeqerr;
    double debugfs;
    double debugff;
    double debugdx;
    ae_int_t debugfeasqpits;
    ae_int_t debugfeasgpaits;
    ae_int_t inneriterationscount;
    ae_int_t outeriterationscount;
} minbleicreport;
#endif
#if defined(AE_COMPILE_QPBLEICSOLVER) || !defined(AE_PARTIAL_BUILD)
typedef struct
{
    double epsg;
    double epsf;
    double epsx;
    ae_int_t maxits;
} qpbleicsettings;
typedef struct
{
    minbleicstate solver;
    minbleicreport solverrep;
    ae_vector tmp0;
    ae_vector tmp1;
    ae_vector tmpi;
    ae_int_t repinneriterationscount;
    ae_int_t repouteriterationscount;
} qpbleicbuffers;
#endif
#if defined(AE_COMPILE_MINQP) || !defined(AE_PARTIAL_BUILD)
typedef struct
{
    ae_int_t n;
    qqpsettings qqpsettingsuser;
    qpbleicsettings qpbleicsettingsuser;
    qpdenseaulsettings qpdenseaulsettingsuser;
    ae_bool dbgskipconstraintnormalization;
    ae_int_t algokind;
    ae_int_t akind;
    convexquadraticmodel a;
    sparsematrix sparsea;
    ae_bool sparseaupper;
    double absamax;
    double absasum;
    double absasum2;
    ae_vector b;
    ae_vector bndl;
    ae_vector bndu;
    ae_int_t stype;
    ae_vector s;
    ae_vector havebndl;
    ae_vector havebndu;
    ae_vector xorigin;
    ae_vector startx;
    ae_bool havex;
    ae_matrix cleic;
    ae_int_t nec;
    ae_int_t nic;
    sparsematrix scleic;
    ae_int_t snec;
    ae_int_t snic;
    ae_vector xs;
    ae_int_t repinneriterationscount;
    ae_int_t repouteriterationscount;
    ae_int_t repncholesky;
    ae_int_t repnmv;
    ae_int_t repterminationtype;
    ae_vector effectives;
    ae_vector tmp0;
    ae_matrix ecleic;
    ae_matrix dummyr2;
    ae_bool qpbleicfirstcall;
    qpbleicbuffers qpbleicbuf;
    qqpbuffers qqpbuf;
    qpdenseaulbuffers qpdenseaulbuf;
} minqpstate;
typedef struct
{
    ae_int_t inneriterationscount;
    ae_int_t outeriterationscount;
    ae_int_t nmv;
    ae_int_t ncholesky;
    ae_int_t terminationtype;
} minqpreport;
#endif
#if defined(AE_COMPILE_REVISEDDUALSIMPLEX) || !defined(AE_PARTIAL_BUILD)
typedef struct
{
    double pivottol;
    double perturbmag;
    ae_int_t maxtrfage;
    ae_int_t trftype;
    ae_int_t ratiotest;
    ae_int_t pricing;
    ae_int_t shifting;
} dualsimplexsettings;
typedef struct
{
    ae_int_t ns;
    ae_int_t m;
    ae_vector idx;
    ae_vector nidx;
    ae_vector isbasic;
    ae_int_t trftype;
    ae_bool isvalidtrf;
    ae_int_t trfage;
    ae_matrix denselu;
    sparsematrix sparsel;
    sparsematrix sparseu;
    sparsematrix sparseut;
    ae_vector rowpermbwd;
    ae_vector colpermbwd;
    ae_vector densepfieta;
    ae_vector densemu;
    ae_vector rk;
    ae_vector dk;
    ae_vector dseweights;
    ae_bool dsevalid;
    double eminu;
    ae_vector wtmp0;
    ae_vector wtmp1;
    ae_vector wtmp2;
    ae_vector nrs;
    ae_vector tcinvidx;
    ae_matrix denselu2;
    ae_vector densep2;
    ae_vector densep2c;
    sparsematrix sparselu1;
    sparsematrix sparselu2;
    sluv2buffer lubuf2;
    ae_vector tmpi;
    ae_vector utmp0;
    ae_vector utmpi;
    sparsematrix sparseludbg;
} dualsimplexbasis;
typedef struct
{
    ae_int_t ns;
    ae_int_t m;
    ae_vector rawc;
    ae_vector bndl;
    ae_vector bndu;
    ae_vector bndt;
    ae_vector xa;
    ae_vector d;
    ae_int_t state;
    ae_vector xb;
    ae_vector bndlb;
    ae_vector bndub;
    ae_vector bndtb;
    ae_vector effc;
    ae_vector colscales;
} dualsimplexsubproblem;
typedef struct
{
    ae_vector varscales;
    ae_vector rowscales;
    ae_vector rawbndl;
    ae_vector rawbndu;
    ae_int_t ns;
    ae_int_t m;
    sparsematrix a;
    sparsematrix at;
    dualsimplexbasis basis;
    dualsimplexsubproblem primary;
    dualsimplexsubproblem phase1;
    dualsimplexsubproblem phase3;
    ae_vector repx;
    ae_vector repy;
    ae_vector repdx;
    ae_vector repstats;
    double repf;
    double repprimalerror;
    double repdualerror;
    ae_int_t repterminationtype;
    ae_int_t repiterationscount;
    ae_int_t repiterationscount1;
    ae_int_t repiterationscount2;
    ae_int_t repiterationscount3;
    ae_vector possibleflips;
    ae_int_t possibleflipscnt;
    ae_vector dfctmp0;
    ae_vector dfctmp1;
    ae_vector dfctmp2;
    ae_vector ustmpi;
    ae_vector tmp0;
    ae_vector tmp1;
    ae_vector tmp2;
    ae_vector alphar;
    ae_vector rhor;
    ae_vector tau;
    ae_vector alphaq;
    ae_vector alphaqim;
    ae_vector eligibleset;
    ae_vector harrisset;
} dualsimplexstate;
#endif
#if defined(AE_COMPILE_MINLP) || !defined(AE_PARTIAL_BUILD)
typedef struct
{
    ae_int_t n;
    ae_int_t algokind;
    ae_vector s;
    ae_vector c;
    ae_vector bndl;
    ae_vector bndu;
    ae_int_t m;
    sparsematrix a;
    ae_vector al;
    ae_vector au;
    ae_vector xs;
    ae_vector ys;
    ae_vector cs;
    double repf;
    double repprimalerror;
    double repdualerror;
    ae_int_t repiterationscount;
    ae_int_t repterminationtype;
    dualsimplexstate dss;
    ae_vector adddtmpi;
    ae_vector adddtmpr;
} minlpstate;
typedef struct
{
    double f;
    ae_vector y;
    ae_vector stats;
    double primalerror;
    double dualerror;
    ae_int_t iterationscount;
    ae_int_t terminationtype;
} minlpreport;
#endif
#if defined(AE_COMPILE_NLCSLP) || !defined(AE_PARTIAL_BUILD)
typedef struct
{
    dualsimplexstate dss;
    dualsimplexsettings dsssettings;
    dualsimplexbasis lastbasis;
    ae_bool basispresent;
    ae_matrix curd;
    ae_int_t curdcnt;
    ae_vector curb;
    ae_vector curbndl;
    ae_vector curbndu;
    ae_vector cural;
    ae_vector curau;
    sparsematrix sparserawlc;
    sparsematrix sparseefflc;
    ae_int_t hessiantype;
    ae_matrix h;
    ae_matrix curhd;
    ae_matrix densedummy;
    sparsematrix sparsedummy;
    ae_vector tmp0;
    ae_vector tmp1;
    ae_vector sk;
    ae_vector yk;
} minslpsubsolver;
typedef struct
{
    ae_int_t n;
    ae_int_t nec;
    ae_int_t nic;
    ae_int_t nlec;
    ae_int_t nlic;
    ae_matrix scaledcleic;
    ae_vector lcsrcidx;
    ae_vector hasbndl;
    ae_vector hasbndu;
    ae_vector scaledbndl;
    ae_vector scaledbndu;
    double epsx;
    ae_int_t maxits;
    ae_int_t hessiantype;
    ae_vector x;
    ae_vector fi;
    ae_matrix j;
    double f;
    ae_bool needfij;
    ae_bool xupdated;
    double trustrad;
    double deltamax;
    minslpsubsolver subsolver;
    ae_vector d;
    ae_vector d0;
    ae_vector d1;
    linminstate mcstate;
    ae_int_t xstagnationcnt;
    ae_int_t fstagnationcnt;
    ae_vector prevx;
    ae_vector step0x;
    ae_vector stepkx;
    ae_vector stepkxc;
    ae_vector stepkxn;
    ae_vector step0fi;
    ae_vector stepkfi;
    ae_vector stepkfic;
    ae_vector stepkfin;
    ae_matrix step0j;
    ae_matrix stepkj;
    ae_matrix stepkjc;
    ae_matrix stepkjn;
    double stepklagval;
    double stepkclagval;
    double stepknlagval;
    ae_vector stepklaggrad;
    ae_vector stepknlaggrad;
    ae_vector stepklagmult;
    ae_vector stepknlagmult;
    ae_vector rho;
    ae_vector tmp0;
    ae_vector sclagtmp0;
    ae_vector sclagtmp1;
    double lastlcerr;
    ae_int_t lastlcidx;
    double lastnlcerr;
    ae_int_t lastnlcidx;
    ae_vector mftmp0;
    ae_int_t repsimplexiterations;
    ae_int_t repsimplexiterations1;
    ae_int_t repsimplexiterations2;
    ae_int_t repsimplexiterations3;
    ae_int_t repinneriterationscount;
    ae_int_t repouteriterationscount;
    ae_int_t repterminationtype;
    double repbcerr;
    ae_int_t repbcidx;
    double replcerr;
    ae_int_t replcidx;
    double repnlcerr;
    ae_int_t repnlcidx;
    rcommstate rstate;
    rcommstate rphase13state;
    rcommstate rphase2state;
} minslpstate;
#endif
#if defined(AE_COMPILE_MINNLC) || !defined(AE_PARTIAL_BUILD)
typedef struct
{
    double stabilizingpoint;
    double initialinequalitymultiplier;
    ae_int_t solvertype;
    ae_int_t prectype;
    ae_int_t updatefreq;
    double rho;
    ae_int_t n;
    double epsx;
    ae_int_t maxits;
    ae_int_t aulitscnt;
    ae_bool xrep;
    double stpmax;
    double diffstep;
    double teststep;
    ae_vector s;
    ae_vector bndl;
    ae_vector bndu;
    ae_vector hasbndl;
    ae_vector hasbndu;
    ae_int_t nec;
    ae_int_t nic;
    ae_matrix cleic;
    ae_vector lcsrcidx;
    ae_int_t ng;
    ae_int_t nh;
    ae_vector x;
    double f;
    ae_vector fi;
    ae_matrix j;
    ae_bool needfij;
    ae_bool needfi;
    ae_bool xupdated;
    rcommstate rstate;
    rcommstate rstateaul;
    rcommstate rstateslp;
    ae_vector scaledbndl;
    ae_vector scaledbndu;
    ae_matrix scaledcleic;
    ae_vector xc;
    ae_vector xstart;
    ae_vector xbase;
    ae_vector fbase;
    ae_vector dfbase;
    ae_vector fm2;
    ae_vector fm1;
    ae_vector fp1;
    ae_vector fp2;
    ae_vector dfm1;
    ae_vector dfp1;
    ae_vector bufd;
    ae_vector bufc;
    ae_vector tmp0;
    ae_matrix bufw;
    ae_matrix bufz;
    ae_vector xk;
    ae_vector xk1;
    ae_vector gk;
    ae_vector gk1;
    double gammak;
    ae_bool xkpresent;
    minlbfgsstate auloptimizer;
    minlbfgsreport aulreport;
    ae_vector nubc;
    ae_vector nulc;
    ae_vector nunlc;
    ae_bool userterminationneeded;
    minslpstate slpsolverstate;
    ae_int_t smoothnessguardlevel;
    smoothnessmonitor smonitor;
    ae_vector lastscaleused;
    ae_int_t repinneriterationscount;
    ae_int_t repouteriterationscount;
    ae_int_t repnfev;
    ae_int_t repterminationtype;
    double repbcerr;
    ae_int_t repbcidx;
    double replcerr;
    ae_int_t replcidx;
    double repnlcerr;
    ae_int_t repnlcidx;
    ae_int_t repdbgphase0its;
} minnlcstate;
typedef struct
{
    ae_int_t iterationscount;
    ae_int_t nfev;
    ae_int_t terminationtype;
    double bcerr;
    ae_int_t bcidx;
    double lcerr;
    ae_int_t lcidx;
    double nlcerr;
    ae_int_t nlcidx;
    ae_int_t dbgphase0its;
} minnlcreport;
#endif
#if defined(AE_COMPILE_MINBC) || !defined(AE_PARTIAL_BUILD)
typedef struct
{
    ae_int_t nmain;
    double epsg;
    double epsf;
    double epsx;
    ae_int_t maxits;
    ae_bool xrep;
    double stpmax;
    double diffstep;
    ae_vector s;
    ae_int_t prectype;
    ae_vector diagh;
    ae_vector x;
    double f;
    ae_vector g;
    ae_bool needf;
    ae_bool needfg;
    ae_bool xupdated;
    ae_bool userterminationneeded;
    rcommstate rstate;
    ae_vector xc;
    ae_vector ugc;
    ae_vector cgc;
    ae_vector xn;
    ae_vector ugn;
    ae_vector cgn;
    ae_vector xp;
    double fc;
    double fn;
    double fp;
    ae_vector d;
    double lastscaledgoodstep;
    ae_vector hasbndl;
    ae_vector hasbndu;
    ae_vector bndl;
    ae_vector bndu;
    ae_int_t repiterationscount;
    ae_int_t repnfev;
    ae_int_t repvaridx;
    ae_int_t repterminationtype;
    ae_vector xstart;
    double fbase;
    double fm2;
    double fm1;
    double fp1;
    double fp2;
    double xm1;
    double xp1;
    double gm1;
    double gp1;
    ae_vector tmpprec;
    ae_vector tmp0;
    ae_int_t nfev;
    ae_int_t mcstage;
    double stp;
    double curstpmax;
    ae_vector work;
    linminstate lstate;
    double trimthreshold;
    ae_int_t nonmonotoniccnt;
    ae_matrix bufyk;
    ae_matrix bufsk;
    ae_vector bufrho;
    ae_vector buftheta;
    ae_int_t bufsize;
    double teststep;
    ae_int_t smoothnessguardlevel;
    smoothnessmonitor smonitor;
    ae_vector lastscaleused;
    ae_vector invs;
} minbcstate;
typedef struct
{
    ae_int_t iterationscount;
    ae_int_t nfev;
    ae_int_t varidx;
    ae_int_t terminationtype;
} minbcreport;
#endif
#if defined(AE_COMPILE_MINNS) || !defined(AE_PARTIAL_BUILD)
typedef struct
{
    double fc;
    double fn;
    ae_vector xc;
    ae_vector xn;
    ae_vector x0;
    ae_vector gc;
    ae_vector d;
    ae_matrix uh;
    ae_matrix ch;
    ae_matrix rk;
    ae_vector invutc;
    ae_vector tmp0;
    ae_vector tmpidx;
    ae_vector tmpd;
    ae_vector tmpc;
    ae_vector tmplambdas;
    ae_matrix tmpc2;
    ae_vector tmpb;
    snnlssolver nnls;
} minnsqp;
typedef struct
{
    ae_int_t solvertype;
    ae_int_t n;
    double epsx;
    ae_int_t maxits;
    ae_bool xrep;
    double diffstep;
    ae_vector s;
    ae_vector bndl;
    ae_vector bndu;
    ae_vector hasbndl;
    ae_vector hasbndu;
    ae_int_t nec;
    ae_int_t nic;
    ae_matrix cleic;
    ae_int_t ng;
    ae_int_t nh;
    ae_vector x;
    double f;
    ae_vector fi;
    ae_matrix j;
    ae_bool needfij;
    ae_bool needfi;
    ae_bool xupdated;
    rcommstate rstate;
    rcommstate rstateags;
    hqrndstate agsrs;
    double agsradius;
    ae_int_t agssamplesize;
    double agsraddecay;
    double agsalphadecay;
    double agsdecrease;
    double agsinitstp;
    double agsstattold;
    double agsshortstpabs;
    double agsshortstprel;
    double agsshortf;
    ae_int_t agsshortlimit;
    double agsrhononlinear;
    ae_int_t agsminupdate;
    ae_int_t agsmaxraddecays;
    ae_int_t agsmaxbacktrack;
    ae_int_t agsmaxbacktracknonfull;
    double agspenaltylevel;
    double agspenaltyincrease;
    ae_vector xstart;
    ae_vector xc;
    ae_vector xn;
    ae_vector grs;
    ae_vector d;
    ae_vector colmax;
    ae_vector diagh;
    ae_vector signmin;
    ae_vector signmax;
    ae_bool userterminationneeded;
    ae_vector scaledbndl;
    ae_vector scaledbndu;
    ae_matrix scaledcleic;
    ae_vector rholinear;
    ae_matrix samplex;
    ae_matrix samplegm;
    ae_matrix samplegmbc;
    ae_vector samplef;
    ae_vector samplef0;
    minnsqp nsqp;
    ae_vector tmp0;
    ae_vector tmp1;
    ae_matrix tmp2;
    ae_vector tmp3;
    ae_vector xbase;
    ae_vector fp;
    ae_vector fm;
    ae_int_t repinneriterationscount;
    ae_int_t repouteriterationscount;
    ae_int_t repnfev;
    ae_int_t repvaridx;
    ae_int_t repfuncidx;
    ae_int_t repterminationtype;
    double replcerr;
    double repnlcerr;
    ae_int_t dbgncholesky;
} minnsstate;
typedef struct
{
    ae_int_t iterationscount;
    ae_int_t nfev;
    double cerr;
    double lcerr;
    double nlcerr;
    ae_int_t terminationtype;
    ae_int_t varidx;
    ae_int_t funcidx;
} minnsreport;
#endif
#if defined(AE_COMPILE_MINCOMP) || !defined(AE_PARTIAL_BUILD)
typedef struct
{
    ae_int_t n;
    double epsg;
    double epsf;
    double epsx;
    ae_int_t maxits;
    ae_bool xrep;
    double stpmax;
    ae_int_t cgtype;
    ae_int_t k;
    ae_int_t nfev;
    ae_int_t mcstage;
    ae_vector bndl;
    ae_vector bndu;
    ae_int_t curalgo;
    ae_int_t acount;
    double mu;
    double finit;
    double dginit;
    ae_vector ak;
    ae_vector xk;
    ae_vector dk;
    ae_vector an;
    ae_vector xn;
    ae_vector dn;
    ae_vector d;
    double fold;
    double stp;
    ae_vector work;
    ae_vector yk;
    ae_vector gc;
    double laststep;
    ae_vector x;
    double f;
    ae_vector g;
    ae_bool needfg;
    ae_bool xupdated;
    rcommstate rstate;
    ae_int_t repiterationscount;
    ae_int_t repnfev;
    ae_int_t repterminationtype;
    ae_int_t debugrestartscount;
    linminstate lstate;
    double betahs;
    double betady;
} minasastate;
typedef struct
{
    ae_int_t iterationscount;
    ae_int_t nfev;
    ae_int_t terminationtype;
    ae_int_t activeconstraints;
} minasareport;
#endif
#if defined(AE_COMPILE_MINCG) || !defined(AE_PARTIAL_BUILD)
typedef struct
{
    ae_int_t n;
    double epsg;
    double epsf;
    double epsx;
    ae_int_t maxits;
    double stpmax;
    double suggestedstep;
    ae_bool xrep;
    ae_bool drep;
    ae_int_t cgtype;
    ae_int_t prectype;
    ae_vector diagh;
    ae_vector diaghl2;
    ae_matrix vcorr;
    ae_int_t vcnt;
    ae_vector s;
    double diffstep;
    ae_int_t nfev;
    ae_int_t mcstage;
    ae_int_t k;
    ae_vector xk;
    ae_vector dk;
    ae_vector xn;
    ae_vector dn;
    ae_vector d;
    double fold;
    double stp;
    double curstpmax;
    ae_vector yk;
    double lastgoodstep;
    double lastscaledstep;
    ae_int_t mcinfo;
    ae_bool innerresetneeded;
    ae_bool terminationneeded;
    double trimthreshold;
    ae_vector xbase;
    ae_int_t rstimer;
    ae_vector x;
    double f;
    ae_vector g;
    ae_bool needf;
    ae_bool needfg;
    ae_bool xupdated;
    ae_bool algpowerup;
    ae_bool lsstart;
    ae_bool lsend;
    ae_bool userterminationneeded;
    rcommstate rstate;
    ae_int_t repiterationscount;
    ae_int_t repnfev;
    ae_int_t repterminationtype;
    ae_int_t debugrestartscount;
    linminstate lstate;
    double fbase;
    double fm2;
    double fm1;
    double fp1;
    double fp2;
    double betahs;
    double betady;
    ae_vector work0;
    ae_vector work1;
    ae_vector invs;
    double teststep;
    ae_int_t smoothnessguardlevel;
    smoothnessmonitor smonitor;
    ae_vector lastscaleused;
} mincgstate;
typedef struct
{
    ae_int_t iterationscount;
    ae_int_t nfev;
    ae_int_t terminationtype;
} mincgreport;
#endif
#if defined(AE_COMPILE_MINLM) || !defined(AE_PARTIAL_BUILD)
typedef struct
{
    ae_int_t n;
    ae_int_t m;
    double stpmax;
    ae_int_t modelage;
    ae_int_t maxmodelage;
    ae_bool hasfi;
    double epsx;
    ae_vector x;
    double f;
    ae_vector fi;
    ae_bool needf;
    ae_bool needfi;
    double fbase;
    ae_vector modeldiag;
    ae_vector xbase;
    ae_vector fibase;
    ae_vector bndl;
    ae_vector bndu;
    ae_vector havebndl;
    ae_vector havebndu;
    ae_vector s;
    rcommstate rstate;
    ae_vector xdir;
    ae_vector choleskybuf;
    ae_vector tmp0;
    ae_vector tmpct;
    double actualdecrease;
    double predicteddecrease;
    minqpstate qpstate;
    minqpreport qprep;
    sparsematrix tmpsp;
} minlmstepfinder;
typedef struct
{
    ae_int_t n;
    ae_int_t m;
    double diffstep;
    double epsx;
    ae_int_t maxits;
    ae_bool xrep;
    double stpmax;
    ae_int_t maxmodelage;
    ae_bool makeadditers;
    ae_vector x;
    double f;
    ae_vector fi;
    ae_matrix j;
    ae_matrix h;
    ae_vector g;
    ae_bool needf;
    ae_bool needfg;
    ae_bool needfgh;
    ae_bool needfij;
    ae_bool needfi;
    ae_bool xupdated;
    ae_bool userterminationneeded;
    ae_int_t algomode;
    ae_bool hasf;
    ae_bool hasfi;
    ae_bool hasg;
    ae_vector xbase;
    double fbase;
    ae_vector fibase;
    ae_vector gbase;
    ae_matrix quadraticmodel;
    ae_vector bndl;
    ae_vector bndu;
    ae_vector havebndl;
    ae_vector havebndu;
    ae_vector s;
    ae_matrix cleic;
    ae_int_t nec;
    ae_int_t nic;
    double lambdav;
    double nu;
    ae_int_t modelage;
    ae_vector xnew;
    ae_vector xdir;
    ae_vector deltax;
    ae_vector deltaf;
    ae_bool deltaxready;
    ae_bool deltafready;
    smoothnessmonitor smonitor;
    double teststep;
    ae_vector lastscaleused;
    ae_int_t repiterationscount;
    ae_int_t repterminationtype;
    ae_int_t repnfunc;
    ae_int_t repnjac;
    ae_int_t repngrad;
    ae_int_t repnhess;
    ae_int_t repncholesky;
    rcommstate rstate;
    ae_vector choleskybuf;
    ae_vector tmp0;
    double actualdecrease;
    double predicteddecrease;
    double xm1;
    double xp1;
    ae_vector fm1;
    ae_vector fp1;
    ae_vector fc1;
    ae_vector gm1;
    ae_vector gp1;
    ae_vector gc1;
    minlbfgsstate internalstate;
    minlbfgsreport internalrep;
    minqpstate qpstate;
    minqpreport qprep;
    minlmstepfinder finderstate;
} minlmstate;
typedef struct
{
    ae_int_t iterationscount;
    ae_int_t terminationtype;
    ae_int_t nfunc;
    ae_int_t njac;
    ae_int_t ngrad;
    ae_int_t nhess;
    ae_int_t ncholesky;
} minlmreport;
#endif

}

/////////////////////////////////////////////////////////////////////////
//
// THIS SECTION CONTAINS C++ INTERFACE
//
/////////////////////////////////////////////////////////////////////////
namespace alglib
{

#if defined(AE_COMPILE_CQMODELS) || !defined(AE_PARTIAL_BUILD)

#endif

#if defined(AE_COMPILE_OPTGUARDAPI) || !defined(AE_PARTIAL_BUILD)
/*************************************************************************
This structure is used to store  OptGuard  report,  i.e.  report  on   the
properties of the nonlinear function being optimized with ALGLIB.

After you tell your optimizer to activate OptGuard  this technology starts
to silently monitor function values and gradients/Jacobians  being  passed
all around during your optimization session. Depending on specific set  of
checks enabled OptGuard may perform additional function evaluations  (say,
about 3*N evaluations if you want to check analytic gradient for errors).

Upon discovering that something strange happens  (function  values  and/or
gradient components change too sharply and/or unexpectedly) OptGuard  sets
one of the "suspicion  flags" (without interrupting optimization session).
After optimization is done, you can examine OptGuard report.

Following report fields can be set:
* nonc0suspected
* nonc1suspected
* badgradsuspected


=== WHAT CAN BE DETECTED WITH OptGuard INTEGRITY CHECKER =================

Following  types  of  errors  in your target function (constraints) can be
caught:
a) discontinuous functions ("non-C0" part of the report)
b) functions with discontinuous derivative ("non-C1" part of the report)
c) errors in the analytic gradient provided by user

These types of errors result in optimizer  stopping  well  before reaching
solution (most often - right after encountering discontinuity).

Type A errors are usually  coding  errors  during  implementation  of  the
target function. Most "normal" problems involve continuous functions,  and
anyway you can't reliably optimize discontinuous function.

Type B errors are either coding errors or (in case code itself is correct)
evidence of the fact  that  your  problem  is  an  "incorrect"  one.  Most
optimizers (except for ones provided by MINNS subpackage) do  not  support
nonsmooth problems.

Type C errors are coding errors which often prevent optimizer from  making
even one step  or result in optimizing stopping  too  early,  as  soon  as
actual descent direction becomes too different from one suggested by user-
supplied gradient.


=== WHAT IS REPORTED =====================================================

Following set of report fields deals with discontinuous  target functions,
ones not belonging to C0 continuity class:

* nonc0suspected - is a flag which is set upon discovering some indication
  of the discontinuity. If this flag is false, the rest of "non-C0" fields
  should be ignored
* nonc0fidx - is an index of the function (0 for  target  function,  1  or
  higher for nonlinear constraints) which is suspected of being "non-C0"
* nonc0lipshitzc - a Lipchitz constant for a function which was  suspected
  of being non-continuous.
* nonc0test0positive -  set  to  indicate  specific  test  which  detected
  continuity violation (test #0)

Following set of report fields deals with discontinuous gradient/Jacobian,
i.e. with functions violating C1 continuity:

* nonc1suspected - is a flag which is set upon discovering some indication
  of the discontinuity. If this flag is false, the rest of "non-C1" fields
  should be ignored
* nonc1fidx - is an index of the function (0 for  target  function,  1  or
  higher for nonlinear constraints) which is suspected of being "non-C1"
* nonc1lipshitzc - a Lipchitz constant for a function gradient  which  was
  suspected of being non-smooth.
* nonc1test0positive -  set  to  indicate  specific  test  which  detected
  continuity violation (test #0)
* nonc1test1positive -  set  to  indicate  specific  test  which  detected
  continuity violation (test #1)

Following set of report fields deals with errors in the gradient:
* badgradsuspected - is a flad which is set upon discovering an  error  in
  the analytic gradient supplied by user
* badgradfidx - index  of   the  function  with bad gradient (0 for target
  function, 1 or higher for nonlinear constraints)
* badgradvidx - index of the variable
* badgradxbase - location where Jacobian is tested
* following  matrices  store  user-supplied  Jacobian  and  its  numerical
  differentiation version (which is assumed to be  free  from  the  coding
  errors), both of them computed near the initial point:
  * badgraduser, an array[K,N], analytic Jacobian supplied by user
  * badgradnum,  an array[K,N], numeric  Jacobian computed by ALGLIB
  Here K is a total number of  nonlinear  functions  (target  +  nonlinear
  constraints), N is a variable number.
  The  element  of  badgraduser[] with index [badgradfidx,badgradvidx]  is
  assumed to be wrong.

More detailed error log can  be  obtained  from  optimizer  by  explicitly
requesting reports for tests C0.0, C1.0, C1.1.

  -- ALGLIB --
     Copyright 19.11.2018 by Bochkanov Sergey
*************************************************************************/
class _optguardreport_owner
{
public:
    _optguardreport_owner();
    _optguardreport_owner(const _optguardreport_owner &rhs);
    _optguardreport_owner& operator=(const _optguardreport_owner &rhs);
    virtual ~_optguardreport_owner();
    alglib_impl::optguardreport* c_ptr();
    alglib_impl::optguardreport* c_ptr() const;
protected:
    alglib_impl::optguardreport *p_struct;
};
class optguardreport : public _optguardreport_owner
{
public:
    optguardreport();
    optguardreport(const optguardreport &rhs);
    optguardreport& operator=(const optguardreport &rhs);
    virtual ~optguardreport();
    ae_bool &nonc0suspected;
    ae_bool &nonc0test0positive;
    ae_int_t &nonc0fidx;
    double &nonc0lipschitzc;
    ae_bool &nonc1suspected;
    ae_bool &nonc1test0positive;
    ae_bool &nonc1test1positive;
    ae_int_t &nonc1fidx;
    double &nonc1lipschitzc;
    ae_bool &badgradsuspected;
    ae_int_t &badgradfidx;
    ae_int_t &badgradvidx;
    real_1d_array badgradxbase;
    real_2d_array badgraduser;
    real_2d_array badgradnum;

};


/*************************************************************************
This  structure  is  used  for  detailed   reporting  about  suspected  C1
continuity violation as flagged by C1 test #0 (OptGuard  has several tests
for C1 continuity, this report is used by #0).

=== WHAT IS TESTED =======================================================

C1 test #0 studies function values (not gradient!)  obtained  during  line
searches and monitors behavior of directional  derivative  estimate.  This
test is less powerful than test #1, but it does  not  depend  on  gradient
values  and  thus  it  is  more  robust  against  artifacts  introduced by
numerical differentiation.


=== WHAT IS REPORTED =====================================================

Actually, report retrieval function returns TWO report structures:

* one for most suspicious point found so far (one with highest  change  in
  the directional derivative), so called "strongest" report
* another one for most detailed line search (more function  evaluations  =
  easier to understand what's going on) which triggered  test #0 criteria,
  so called "longest" report

In both cases following fields are returned:

* positive - is TRUE  when test flagged suspicious point;  FALSE  if  test
  did not notice anything (in the latter cases fields below are empty).
* fidx - is an index of the function (0 for  target  function, 1 or higher
  for nonlinear constraints) which is suspected of being "non-C1"
* x0[], d[] - arrays of length N which store initial point  and  direction
  for line search (d[] can be normalized, but does not have to)
* stp[], f[] - arrays of length CNT which store step lengths and  function
  values at these points; f[i] is evaluated in x0+stp[i]*d.
* stpidxa, stpidxb - we  suspect  that  function  violates  C1  continuity
  between steps #stpidxa and #stpidxb (usually we have  stpidxb=stpidxa+3,
  with  most  likely  position  of  the  violation  between  stpidxa+1 and
  stpidxa+2.

You can plot function values stored in stp[]  and  f[]  arrays  and  study
behavior of your function by your own eyes, just  to  be  sure  that  test
correctly reported C1 violation.

  -- ALGLIB --
     Copyright 19.11.2018 by Bochkanov Sergey
*************************************************************************/
class _optguardnonc1test0report_owner
{
public:
    _optguardnonc1test0report_owner();
    _optguardnonc1test0report_owner(const _optguardnonc1test0report_owner &rhs);
    _optguardnonc1test0report_owner& operator=(const _optguardnonc1test0report_owner &rhs);
    virtual ~_optguardnonc1test0report_owner();
    alglib_impl::optguardnonc1test0report* c_ptr();
    alglib_impl::optguardnonc1test0report* c_ptr() const;
protected:
    alglib_impl::optguardnonc1test0report *p_struct;
};
class optguardnonc1test0report : public _optguardnonc1test0report_owner
{
public:
    optguardnonc1test0report();
    optguardnonc1test0report(const optguardnonc1test0report &rhs);
    optguardnonc1test0report& operator=(const optguardnonc1test0report &rhs);
    virtual ~optguardnonc1test0report();
    ae_bool &positive;
    ae_int_t &fidx;
    real_1d_array x0;
    real_1d_array d;
    ae_int_t &n;
    real_1d_array stp;
    real_1d_array f;
    ae_int_t &cnt;
    ae_int_t &stpidxa;
    ae_int_t &stpidxb;

};


/*************************************************************************
This  structure  is  used  for  detailed   reporting  about  suspected  C1
continuity violation as flagged by C1 test #1 (OptGuard  has several tests
for C1 continuity, this report is used by #1).

=== WHAT IS TESTED =======================================================

C1 test #1 studies individual  components  of  the  gradient  as  recorded
during line searches. Upon discovering discontinuity in the gradient  this
test records specific component which was suspected (or  one  with  highest
indication of discontinuity if multiple components are suspected).

When precise analytic gradient is provided this test is more powerful than
test #0  which  works  with  function  values  and  ignores  user-provided
gradient.  However,  test  #0  becomes  more   powerful   when   numerical
differentiation is employed (in such cases test #1 detects  higher  levels
of numerical noise and becomes too conservative).

This test also tells specific components of the gradient which violate  C1
continuity, which makes it more informative than #0, which just tells that
continuity is violated.


=== WHAT IS REPORTED =====================================================

Actually, report retrieval function returns TWO report structures:

* one for most suspicious point found so far (one with highest  change  in
  the directional derivative), so called "strongest" report
* another one for most detailed line search (more function  evaluations  =
  easier to understand what's going on) which triggered  test #1 criteria,
  so called "longest" report

In both cases following fields are returned:

* positive - is TRUE  when test flagged suspicious point;  FALSE  if  test
  did not notice anything (in the latter cases fields below are empty).
* fidx - is an index of the function (0 for  target  function, 1 or higher
  for nonlinear constraints) which is suspected of being "non-C1"
* vidx - is an index of the variable in [0,N) with nonsmooth derivative
* x0[], d[] - arrays of length N which store initial point  and  direction
  for line search (d[] can be normalized, but does not have to)
* stp[], g[] - arrays of length CNT which store step lengths and  gradient
  values at these points; g[i] is evaluated in  x0+stp[i]*d  and  contains
  vidx-th component of the gradient.
* stpidxa, stpidxb - we  suspect  that  function  violates  C1  continuity
  between steps #stpidxa and #stpidxb (usually we have  stpidxb=stpidxa+3,
  with  most  likely  position  of  the  violation  between  stpidxa+1 and
  stpidxa+2.

You can plot function values stored in stp[]  and  g[]  arrays  and  study
behavior of your function by your own eyes, just  to  be  sure  that  test
correctly reported C1 violation.

  -- ALGLIB --
     Copyright 19.11.2018 by Bochkanov Sergey
*************************************************************************/
class _optguardnonc1test1report_owner
{
public:
    _optguardnonc1test1report_owner();
    _optguardnonc1test1report_owner(const _optguardnonc1test1report_owner &rhs);
    _optguardnonc1test1report_owner& operator=(const _optguardnonc1test1report_owner &rhs);
    virtual ~_optguardnonc1test1report_owner();
    alglib_impl::optguardnonc1test1report* c_ptr();
    alglib_impl::optguardnonc1test1report* c_ptr() const;
protected:
    alglib_impl::optguardnonc1test1report *p_struct;
};
class optguardnonc1test1report : public _optguardnonc1test1report_owner
{
public:
    optguardnonc1test1report();
    optguardnonc1test1report(const optguardnonc1test1report &rhs);
    optguardnonc1test1report& operator=(const optguardnonc1test1report &rhs);
    virtual ~optguardnonc1test1report();
    ae_bool &positive;
    ae_int_t &fidx;
    ae_int_t &vidx;
    real_1d_array x0;
    real_1d_array d;
    ae_int_t &n;
    real_1d_array stp;
    real_1d_array g;
    ae_int_t &cnt;
    ae_int_t &stpidxa;
    ae_int_t &stpidxb;

};
#endif

#if defined(AE_COMPILE_OPTSERV) || !defined(AE_PARTIAL_BUILD)

#endif

#if defined(AE_COMPILE_SNNLS) || !defined(AE_PARTIAL_BUILD)

#endif

#if defined(AE_COMPILE_SACTIVESETS) || !defined(AE_PARTIAL_BUILD)

#endif

#if defined(AE_COMPILE_QQPSOLVER) || !defined(AE_PARTIAL_BUILD)

#endif

#if defined(AE_COMPILE_MINLBFGS) || !defined(AE_PARTIAL_BUILD)
/*************************************************************************

*************************************************************************/
class _minlbfgsstate_owner
{
public:
    _minlbfgsstate_owner();
    _minlbfgsstate_owner(const _minlbfgsstate_owner &rhs);
    _minlbfgsstate_owner& operator=(const _minlbfgsstate_owner &rhs);
    virtual ~_minlbfgsstate_owner();
    alglib_impl::minlbfgsstate* c_ptr();
    alglib_impl::minlbfgsstate* c_ptr() const;
protected:
    alglib_impl::minlbfgsstate *p_struct;
};
class minlbfgsstate : public _minlbfgsstate_owner
{
public:
    minlbfgsstate();
    minlbfgsstate(const minlbfgsstate &rhs);
    minlbfgsstate& operator=(const minlbfgsstate &rhs);
    virtual ~minlbfgsstate();
    ae_bool &needf;
    ae_bool &needfg;
    ae_bool &xupdated;
    double &f;
    real_1d_array g;
    real_1d_array x;

};


/*************************************************************************
This structure stores optimization report:
* IterationsCount           total number of inner iterations
* NFEV                      number of gradient evaluations
* TerminationType           termination type (see below)

TERMINATION CODES

TerminationType field contains completion code, which can be:
  -8    internal integrity control detected  infinite  or  NAN  values  in
        function/gradient. Abnormal termination signalled.
   1    relative function improvement is no more than EpsF.
   2    relative step is no more than EpsX.
   4    gradient norm is no more than EpsG
   5    MaxIts steps was taken
   7    stopping conditions are too stringent,
        further improvement is impossible,
        X contains best point found so far.
   8    terminated    by  user  who  called  minlbfgsrequesttermination().
        X contains point which was   "current accepted"  when  termination
        request was submitted.

Other fields of this structure are not documented and should not be used!
*************************************************************************/
class _minlbfgsreport_owner
{
public:
    _minlbfgsreport_owner();
    _minlbfgsreport_owner(const _minlbfgsreport_owner &rhs);
    _minlbfgsreport_owner& operator=(const _minlbfgsreport_owner &rhs);
    virtual ~_minlbfgsreport_owner();
    alglib_impl::minlbfgsreport* c_ptr();
    alglib_impl::minlbfgsreport* c_ptr() const;
protected:
    alglib_impl::minlbfgsreport *p_struct;
};
class minlbfgsreport : public _minlbfgsreport_owner
{
public:
    minlbfgsreport();
    minlbfgsreport(const minlbfgsreport &rhs);
    minlbfgsreport& operator=(const minlbfgsreport &rhs);
    virtual ~minlbfgsreport();
    ae_int_t &iterationscount;
    ae_int_t &nfev;
    ae_int_t &terminationtype;

};
#endif

#if defined(AE_COMPILE_QPDENSEAULSOLVER) || !defined(AE_PARTIAL_BUILD)

#endif

#if defined(AE_COMPILE_MINBLEIC) || !defined(AE_PARTIAL_BUILD)
/*************************************************************************
This object stores nonlinear optimizer state.
You should use functions provided by MinBLEIC subpackage to work with this
object
*************************************************************************/
class _minbleicstate_owner
{
public:
    _minbleicstate_owner();
    _minbleicstate_owner(const _minbleicstate_owner &rhs);
    _minbleicstate_owner& operator=(const _minbleicstate_owner &rhs);
    virtual ~_minbleicstate_owner();
    alglib_impl::minbleicstate* c_ptr();
    alglib_impl::minbleicstate* c_ptr() const;
protected:
    alglib_impl::minbleicstate *p_struct;
};
class minbleicstate : public _minbleicstate_owner
{
public:
    minbleicstate();
    minbleicstate(const minbleicstate &rhs);
    minbleicstate& operator=(const minbleicstate &rhs);
    virtual ~minbleicstate();
    ae_bool &needf;
    ae_bool &needfg;
    ae_bool &xupdated;
    double &f;
    real_1d_array g;
    real_1d_array x;

};


/*************************************************************************
This structure stores optimization report:
* IterationsCount           number of iterations
* NFEV                      number of gradient evaluations
* TerminationType           termination type (see below)

TERMINATION CODES

TerminationType field contains completion code, which can be:
  -8    internal integrity control detected  infinite  or  NAN  values  in
        function/gradient. Abnormal termination signalled.
  -3    inconsistent constraints. Feasible point is
        either nonexistent or too hard to find. Try to
        restart optimizer with better initial approximation
   1    relative function improvement is no more than EpsF.
   2    relative step is no more than EpsX.
   4    gradient norm is no more than EpsG
   5    MaxIts steps was taken
   7    stopping conditions are too stringent,
        further improvement is impossible,
        X contains best point found so far.
   8    terminated by user who called minbleicrequesttermination(). X contains
        point which was "current accepted" when  termination  request  was
        submitted.

ADDITIONAL FIELDS

There are additional fields which can be used for debugging:
* DebugEqErr                error in the equality constraints (2-norm)
* DebugFS                   f, calculated at projection of initial point
                            to the feasible set
* DebugFF                   f, calculated at the final point
* DebugDX                   |X_start-X_final|
*************************************************************************/
class _minbleicreport_owner
{
public:
    _minbleicreport_owner();
    _minbleicreport_owner(const _minbleicreport_owner &rhs);
    _minbleicreport_owner& operator=(const _minbleicreport_owner &rhs);
    virtual ~_minbleicreport_owner();
    alglib_impl::minbleicreport* c_ptr();
    alglib_impl::minbleicreport* c_ptr() const;
protected:
    alglib_impl::minbleicreport *p_struct;
};
class minbleicreport : public _minbleicreport_owner
{
public:
    minbleicreport();
    minbleicreport(const minbleicreport &rhs);
    minbleicreport& operator=(const minbleicreport &rhs);
    virtual ~minbleicreport();
    ae_int_t &iterationscount;
    ae_int_t &nfev;
    ae_int_t &varidx;
    ae_int_t &terminationtype;
    double &debugeqerr;
    double &debugfs;
    double &debugff;
    double &debugdx;
    ae_int_t &debugfeasqpits;
    ae_int_t &debugfeasgpaits;
    ae_int_t &inneriterationscount;
    ae_int_t &outeriterationscount;

};
#endif

#if defined(AE_COMPILE_QPBLEICSOLVER) || !defined(AE_PARTIAL_BUILD)

#endif

#if defined(AE_COMPILE_MINQP) || !defined(AE_PARTIAL_BUILD)
/*************************************************************************
This object stores nonlinear optimizer state.
You should use functions provided by MinQP subpackage to work with this
object
*************************************************************************/
class _minqpstate_owner
{
public:
    _minqpstate_owner();
    _minqpstate_owner(const _minqpstate_owner &rhs);
    _minqpstate_owner& operator=(const _minqpstate_owner &rhs);
    virtual ~_minqpstate_owner();
    alglib_impl::minqpstate* c_ptr();
    alglib_impl::minqpstate* c_ptr() const;
protected:
    alglib_impl::minqpstate *p_struct;
};
class minqpstate : public _minqpstate_owner
{
public:
    minqpstate();
    minqpstate(const minqpstate &rhs);
    minqpstate& operator=(const minqpstate &rhs);
    virtual ~minqpstate();

};


/*************************************************************************
This structure stores optimization report:
* InnerIterationsCount      number of inner iterations
* OuterIterationsCount      number of outer iterations
* NCholesky                 number of Cholesky decomposition
* NMV                       number of matrix-vector products
                            (only products calculated as part of iterative
                            process are counted)
* TerminationType           completion code (see below)

Completion codes:
* -9    failure of the automatic scale evaluation:  one  of  the  diagonal
        elements of the quadratic term is non-positive.  Specify  variable
        scales manually!
* -5    inappropriate solver was used:
        * QuickQP solver for problem with general linear constraints (dense/sparse)
* -4    BLEIC-QP or QuickQP solver found unconstrained direction
        of negative curvature (function is unbounded from
        below  even  under  constraints),  no  meaningful
        minimum can be found.
* -3    inconsistent constraints (or, maybe, feasible point is
        too hard to find). If you are sure that constraints are feasible,
        try to restart optimizer with better initial approximation.
* -1    solver error
*  1..4 successful completion
*  5    MaxIts steps was taken
*  7    stopping conditions are too stringent,
        further improvement is impossible,
        X contains best point found so far.
*************************************************************************/
class _minqpreport_owner
{
public:
    _minqpreport_owner();
    _minqpreport_owner(const _minqpreport_owner &rhs);
    _minqpreport_owner& operator=(const _minqpreport_owner &rhs);
    virtual ~_minqpreport_owner();
    alglib_impl::minqpreport* c_ptr();
    alglib_impl::minqpreport* c_ptr() const;
protected:
    alglib_impl::minqpreport *p_struct;
};
class minqpreport : public _minqpreport_owner
{
public:
    minqpreport();
    minqpreport(const minqpreport &rhs);
    minqpreport& operator=(const minqpreport &rhs);
    virtual ~minqpreport();
    ae_int_t &inneriterationscount;
    ae_int_t &outeriterationscount;
    ae_int_t &nmv;
    ae_int_t &ncholesky;
    ae_int_t &terminationtype;

};
#endif

#if defined(AE_COMPILE_REVISEDDUALSIMPLEX) || !defined(AE_PARTIAL_BUILD)

#endif

#if defined(AE_COMPILE_MINLP) || !defined(AE_PARTIAL_BUILD)
/*************************************************************************
This object stores linear solver state.
You should use functions provided by MinLP subpackage to work with this
object
*************************************************************************/
class _minlpstate_owner
{
public:
    _minlpstate_owner();
    _minlpstate_owner(const _minlpstate_owner &rhs);
    _minlpstate_owner& operator=(const _minlpstate_owner &rhs);
    virtual ~_minlpstate_owner();
    alglib_impl::minlpstate* c_ptr();
    alglib_impl::minlpstate* c_ptr() const;
protected:
    alglib_impl::minlpstate *p_struct;
};
class minlpstate : public _minlpstate_owner
{
public:
    minlpstate();
    minlpstate(const minlpstate &rhs);
    minlpstate& operator=(const minlpstate &rhs);
    virtual ~minlpstate();

};


/*************************************************************************
This structure stores optimization report:
* f                         target function value
* y                         dual variables
* stats                     array[N+M], statuses of box (N) and linear (M)
                            constraints:
                            * stats[i]>0  =>  constraint at upper bound
                                              (also used for free non-basic
                                              variables set to zero)
                            * stats[i]<0  =>  constraint at lower bound
                            * stats[i]=0  =>  constraint is inactive, basic
                                              variable
* primalerror               primal feasibility error
* dualerror                 dual feasibility error
* iterationscount           iteration count
* terminationtype           completion code (see below)

Completion codes:
* -4    LP problem is primal unbounded (dual infeasible)
* -3    LP problem is primal infeasible (dual unbounded)
*  1..4 successful completion
*  5    MaxIts steps was taken
*  7    stopping conditions are too stringent,
        further improvement is impossible,
        X contains best point found so far.
*************************************************************************/
class _minlpreport_owner
{
public:
    _minlpreport_owner();
    _minlpreport_owner(const _minlpreport_owner &rhs);
    _minlpreport_owner& operator=(const _minlpreport_owner &rhs);
    virtual ~_minlpreport_owner();
    alglib_impl::minlpreport* c_ptr();
    alglib_impl::minlpreport* c_ptr() const;
protected:
    alglib_impl::minlpreport *p_struct;
};
class minlpreport : public _minlpreport_owner
{
public:
    minlpreport();
    minlpreport(const minlpreport &rhs);
    minlpreport& operator=(const minlpreport &rhs);
    virtual ~minlpreport();
    double &f;
    real_1d_array y;
    integer_1d_array stats;
    double &primalerror;
    double &dualerror;
    ae_int_t &iterationscount;
    ae_int_t &terminationtype;

};
#endif

#if defined(AE_COMPILE_NLCSLP) || !defined(AE_PARTIAL_BUILD)

#endif

#if defined(AE_COMPILE_MINNLC) || !defined(AE_PARTIAL_BUILD)
/*************************************************************************
This object stores nonlinear optimizer state.
You should use functions provided by MinNLC subpackage to work  with  this
object
*************************************************************************/
class _minnlcstate_owner
{
public:
    _minnlcstate_owner();
    _minnlcstate_owner(const _minnlcstate_owner &rhs);
    _minnlcstate_owner& operator=(const _minnlcstate_owner &rhs);
    virtual ~_minnlcstate_owner();
    alglib_impl::minnlcstate* c_ptr();
    alglib_impl::minnlcstate* c_ptr() const;
protected:
    alglib_impl::minnlcstate *p_struct;
};
class minnlcstate : public _minnlcstate_owner
{
public:
    minnlcstate();
    minnlcstate(const minnlcstate &rhs);
    minnlcstate& operator=(const minnlcstate &rhs);
    virtual ~minnlcstate();
    ae_bool &needfi;
    ae_bool &needfij;
    ae_bool &xupdated;
    double &f;
    real_1d_array fi;
    real_2d_array j;
    real_1d_array x;

};


/*************************************************************************
These fields store optimization report:
* iterationscount           total number of inner iterations
* nfev                      number of gradient evaluations
* terminationtype           termination type (see below)

Scaled constraint violations are reported:
* bcerr                     maximum violation of the box constraints
* bcidx                     index of the most violated box  constraint (or
                            -1, if all box constraints  are  satisfied  or
                            there is no box constraint)
* lcerr                     maximum violation of the  linear  constraints,
                            computed as maximum  scaled  distance  between
                            final point and constraint boundary.
* lcidx                     index of the most violated  linear  constraint
                            (or -1, if all constraints  are  satisfied  or
                            there is no general linear constraints)
* nlcerr                    maximum violation of the nonlinear constraints
* nlcidx                    index of the most violated nonlinear constraint
                            (or -1, if all constraints  are  satisfied  or
                            there is no nonlinear constraints)

Violations of box constraints are scaled on per-component basis  according
to  the  scale  vector s[] as specified by minnlcsetscale(). Violations of
the general linear  constraints  are  also  computed  using  user-supplied
variable scaling. Violations of nonlinear constraints are computed "as is"

TERMINATION CODES

TerminationType field contains completion code, which can be either:

=== FAILURE CODE ===
  -8    internal integrity control detected  infinite  or  NAN  values  in
        function/gradient. Abnormal termination signaled.
  -3    box  constraints  are  infeasible.  Note: infeasibility of non-box
        constraints does NOT trigger emergency  completion;  you  have  to
        examine  bcerr/lcerr/nlcerr   to  detect   possibly   inconsistent
        constraints.

=== SUCCESS CODE ===
   2    relative step is no more than EpsX.
   5    MaxIts steps was taken
   7    stopping conditions are too stringent,
        further improvement is impossible,
        X contains best point found so far.
   8    user requested algorithm termination via minnlcrequesttermination(),
        last accepted point is returned

Other fields of this structure are not documented and should not be used!
*************************************************************************/
class _minnlcreport_owner
{
public:
    _minnlcreport_owner();
    _minnlcreport_owner(const _minnlcreport_owner &rhs);
    _minnlcreport_owner& operator=(const _minnlcreport_owner &rhs);
    virtual ~_minnlcreport_owner();
    alglib_impl::minnlcreport* c_ptr();
    alglib_impl::minnlcreport* c_ptr() const;
protected:
    alglib_impl::minnlcreport *p_struct;
};
class minnlcreport : public _minnlcreport_owner
{
public:
    minnlcreport();
    minnlcreport(const minnlcreport &rhs);
    minnlcreport& operator=(const minnlcreport &rhs);
    virtual ~minnlcreport();
    ae_int_t &iterationscount;
    ae_int_t &nfev;
    ae_int_t &terminationtype;
    double &bcerr;
    ae_int_t &bcidx;
    double &lcerr;
    ae_int_t &lcidx;
    double &nlcerr;
    ae_int_t &nlcidx;
    ae_int_t &dbgphase0its;

};
#endif

#if defined(AE_COMPILE_MINBC) || !defined(AE_PARTIAL_BUILD)
/*************************************************************************
This object stores nonlinear optimizer state.
You should use functions provided by MinBC subpackage to work with this
object
*************************************************************************/
class _minbcstate_owner
{
public:
    _minbcstate_owner();
    _minbcstate_owner(const _minbcstate_owner &rhs);
    _minbcstate_owner& operator=(const _minbcstate_owner &rhs);
    virtual ~_minbcstate_owner();
    alglib_impl::minbcstate* c_ptr();
    alglib_impl::minbcstate* c_ptr() const;
protected:
    alglib_impl::minbcstate *p_struct;
};
class minbcstate : public _minbcstate_owner
{
public:
    minbcstate();
    minbcstate(const minbcstate &rhs);
    minbcstate& operator=(const minbcstate &rhs);
    virtual ~minbcstate();
    ae_bool &needf;
    ae_bool &needfg;
    ae_bool &xupdated;
    double &f;
    real_1d_array g;
    real_1d_array x;

};


/*************************************************************************
This structure stores optimization report:
* iterationscount           number of iterations
* nfev                      number of gradient evaluations
* terminationtype           termination type (see below)

TERMINATION CODES

terminationtype field contains completion code, which can be:
  -8    internal integrity control detected  infinite  or  NAN  values  in
        function/gradient. Abnormal termination signalled.
  -3    inconsistent constraints.
   1    relative function improvement is no more than EpsF.
   2    relative step is no more than EpsX.
   4    gradient norm is no more than EpsG
   5    MaxIts steps was taken
   7    stopping conditions are too stringent,
        further improvement is impossible,
        X contains best point found so far.
   8    terminated by user who called minbcrequesttermination(). X contains
        point which was "current accepted" when  termination  request  was
        submitted.
*************************************************************************/
class _minbcreport_owner
{
public:
    _minbcreport_owner();
    _minbcreport_owner(const _minbcreport_owner &rhs);
    _minbcreport_owner& operator=(const _minbcreport_owner &rhs);
    virtual ~_minbcreport_owner();
    alglib_impl::minbcreport* c_ptr();
    alglib_impl::minbcreport* c_ptr() const;
protected:
    alglib_impl::minbcreport *p_struct;
};
class minbcreport : public _minbcreport_owner
{
public:
    minbcreport();
    minbcreport(const minbcreport &rhs);
    minbcreport& operator=(const minbcreport &rhs);
    virtual ~minbcreport();
    ae_int_t &iterationscount;
    ae_int_t &nfev;
    ae_int_t &varidx;
    ae_int_t &terminationtype;

};
#endif

#if defined(AE_COMPILE_MINNS) || !defined(AE_PARTIAL_BUILD)
/*************************************************************************
This object stores nonlinear optimizer state.
You should use functions provided by MinNS subpackage to work  with  this
object
*************************************************************************/
class _minnsstate_owner
{
public:
    _minnsstate_owner();
    _minnsstate_owner(const _minnsstate_owner &rhs);
    _minnsstate_owner& operator=(const _minnsstate_owner &rhs);
    virtual ~_minnsstate_owner();
    alglib_impl::minnsstate* c_ptr();
    alglib_impl::minnsstate* c_ptr() const;
protected:
    alglib_impl::minnsstate *p_struct;
};
class minnsstate : public _minnsstate_owner
{
public:
    minnsstate();
    minnsstate(const minnsstate &rhs);
    minnsstate& operator=(const minnsstate &rhs);
    virtual ~minnsstate();
    ae_bool &needfi;
    ae_bool &needfij;
    ae_bool &xupdated;
    double &f;
    real_1d_array fi;
    real_2d_array j;
    real_1d_array x;

};


/*************************************************************************
This structure stores optimization report:
* IterationsCount           total number of inner iterations
* NFEV                      number of gradient evaluations
* TerminationType           termination type (see below)
* CErr                      maximum violation of all types of constraints
* LCErr                     maximum violation of linear constraints
* NLCErr                    maximum violation of nonlinear constraints

TERMINATION CODES

TerminationType field contains completion code, which can be:
  -8    internal integrity control detected  infinite  or  NAN  values  in
        function/gradient. Abnormal termination signalled.
  -3    box constraints are inconsistent
  -1    inconsistent parameters were passed:
        * penalty parameter for minnssetalgoags() is zero,
          but we have nonlinear constraints set by minnssetnlc()
   2    sampling radius decreased below epsx
   5    MaxIts steps was taken
   7    stopping conditions are too stringent,
        further improvement is impossible,
        X contains best point found so far.
   8    User requested termination via MinNSRequestTermination()

Other fields of this structure are not documented and should not be used!
*************************************************************************/
class _minnsreport_owner
{
public:
    _minnsreport_owner();
    _minnsreport_owner(const _minnsreport_owner &rhs);
    _minnsreport_owner& operator=(const _minnsreport_owner &rhs);
    virtual ~_minnsreport_owner();
    alglib_impl::minnsreport* c_ptr();
    alglib_impl::minnsreport* c_ptr() const;
protected:
    alglib_impl::minnsreport *p_struct;
};
class minnsreport : public _minnsreport_owner
{
public:
    minnsreport();
    minnsreport(const minnsreport &rhs);
    minnsreport& operator=(const minnsreport &rhs);
    virtual ~minnsreport();
    ae_int_t &iterationscount;
    ae_int_t &nfev;
    double &cerr;
    double &lcerr;
    double &nlcerr;
    ae_int_t &terminationtype;
    ae_int_t &varidx;
    ae_int_t &funcidx;

};
#endif

#if defined(AE_COMPILE_MINCOMP) || !defined(AE_PARTIAL_BUILD)
/*************************************************************************

*************************************************************************/
class _minasastate_owner
{
public:
    _minasastate_owner();
    _minasastate_owner(const _minasastate_owner &rhs);
    _minasastate_owner& operator=(const _minasastate_owner &rhs);
    virtual ~_minasastate_owner();
    alglib_impl::minasastate* c_ptr();
    alglib_impl::minasastate* c_ptr() const;
protected:
    alglib_impl::minasastate *p_struct;
};
class minasastate : public _minasastate_owner
{
public:
    minasastate();
    minasastate(const minasastate &rhs);
    minasastate& operator=(const minasastate &rhs);
    virtual ~minasastate();
    ae_bool &needfg;
    ae_bool &xupdated;
    double &f;
    real_1d_array g;
    real_1d_array x;

};


/*************************************************************************

*************************************************************************/
class _minasareport_owner
{
public:
    _minasareport_owner();
    _minasareport_owner(const _minasareport_owner &rhs);
    _minasareport_owner& operator=(const _minasareport_owner &rhs);
    virtual ~_minasareport_owner();
    alglib_impl::minasareport* c_ptr();
    alglib_impl::minasareport* c_ptr() const;
protected:
    alglib_impl::minasareport *p_struct;
};
class minasareport : public _minasareport_owner
{
public:
    minasareport();
    minasareport(const minasareport &rhs);
    minasareport& operator=(const minasareport &rhs);
    virtual ~minasareport();
    ae_int_t &iterationscount;
    ae_int_t &nfev;
    ae_int_t &terminationtype;
    ae_int_t &activeconstraints;

};
#endif

#if defined(AE_COMPILE_MINCG) || !defined(AE_PARTIAL_BUILD)
/*************************************************************************
This object stores state of the nonlinear CG optimizer.

You should use ALGLIB functions to work with this object.
*************************************************************************/
class _mincgstate_owner
{
public:
    _mincgstate_owner();
    _mincgstate_owner(const _mincgstate_owner &rhs);
    _mincgstate_owner& operator=(const _mincgstate_owner &rhs);
    virtual ~_mincgstate_owner();
    alglib_impl::mincgstate* c_ptr();
    alglib_impl::mincgstate* c_ptr() const;
protected:
    alglib_impl::mincgstate *p_struct;
};
class mincgstate : public _mincgstate_owner
{
public:
    mincgstate();
    mincgstate(const mincgstate &rhs);
    mincgstate& operator=(const mincgstate &rhs);
    virtual ~mincgstate();
    ae_bool &needf;
    ae_bool &needfg;
    ae_bool &xupdated;
    double &f;
    real_1d_array g;
    real_1d_array x;

};


/*************************************************************************
This structure stores optimization report:
* IterationsCount           total number of inner iterations
* NFEV                      number of gradient evaluations
* TerminationType           termination type (see below)

TERMINATION CODES

TerminationType field contains completion code, which can be:
  -8    internal integrity control detected  infinite  or  NAN  values  in
        function/gradient. Abnormal termination signalled.
   1    relative function improvement is no more than EpsF.
   2    relative step is no more than EpsX.
   4    gradient norm is no more than EpsG
   5    MaxIts steps was taken
   7    stopping conditions are too stringent,
        further improvement is impossible,
        X contains best point found so far.
   8    terminated by user who called mincgrequesttermination(). X contains
        point which was "current accepted" when  termination  request  was
        submitted.

Other fields of this structure are not documented and should not be used!
*************************************************************************/
class _mincgreport_owner
{
public:
    _mincgreport_owner();
    _mincgreport_owner(const _mincgreport_owner &rhs);
    _mincgreport_owner& operator=(const _mincgreport_owner &rhs);
    virtual ~_mincgreport_owner();
    alglib_impl::mincgreport* c_ptr();
    alglib_impl::mincgreport* c_ptr() const;
protected:
    alglib_impl::mincgreport *p_struct;
};
class mincgreport : public _mincgreport_owner
{
public:
    mincgreport();
    mincgreport(const mincgreport &rhs);
    mincgreport& operator=(const mincgreport &rhs);
    virtual ~mincgreport();
    ae_int_t &iterationscount;
    ae_int_t &nfev;
    ae_int_t &terminationtype;

};
#endif

#if defined(AE_COMPILE_MINLM) || !defined(AE_PARTIAL_BUILD)
/*************************************************************************
Levenberg-Marquardt optimizer.

This structure should be created using one of the MinLMCreate???()
functions. You should not access its fields directly; use ALGLIB functions
to work with it.
*************************************************************************/
class _minlmstate_owner
{
public:
    _minlmstate_owner();
    _minlmstate_owner(const _minlmstate_owner &rhs);
    _minlmstate_owner& operator=(const _minlmstate_owner &rhs);
    virtual ~_minlmstate_owner();
    alglib_impl::minlmstate* c_ptr();
    alglib_impl::minlmstate* c_ptr() const;
protected:
    alglib_impl::minlmstate *p_struct;
};
class minlmstate : public _minlmstate_owner
{
public:
    minlmstate();
    minlmstate(const minlmstate &rhs);
    minlmstate& operator=(const minlmstate &rhs);
    virtual ~minlmstate();
    ae_bool &needf;
    ae_bool &needfg;
    ae_bool &needfgh;
    ae_bool &needfi;
    ae_bool &needfij;
    ae_bool &xupdated;
    double &f;
    real_1d_array fi;
    real_1d_array g;
    real_2d_array h;
    real_2d_array j;
    real_1d_array x;

};


/*************************************************************************
Optimization report, filled by MinLMResults() function

FIELDS:
* TerminationType, completetion code:
    * -8    optimizer detected NAN/INF values either in the function itself,
            or in its Jacobian
    * -5    inappropriate solver was used:
            * solver created with minlmcreatefgh() used  on  problem  with
              general linear constraints (set with minlmsetlc() call).
    * -3    constraints are inconsistent
    *  2    relative step is no more than EpsX.
    *  5    MaxIts steps was taken
    *  7    stopping conditions are too stringent,
            further improvement is impossible
    *  8    terminated   by  user  who  called  MinLMRequestTermination().
            X contains point which was "current accepted" when termination
            request was submitted.
* IterationsCount, contains iterations count
* NFunc, number of function calculations
* NJac, number of Jacobi matrix calculations
* NGrad, number of gradient calculations
* NHess, number of Hessian calculations
* NCholesky, number of Cholesky decomposition calculations
*************************************************************************/
class _minlmreport_owner
{
public:
    _minlmreport_owner();
    _minlmreport_owner(const _minlmreport_owner &rhs);
    _minlmreport_owner& operator=(const _minlmreport_owner &rhs);
    virtual ~_minlmreport_owner();
    alglib_impl::minlmreport* c_ptr();
    alglib_impl::minlmreport* c_ptr() const;
protected:
    alglib_impl::minlmreport *p_struct;
};
class minlmreport : public _minlmreport_owner
{
public:
    minlmreport();
    minlmreport(const minlmreport &rhs);
    minlmreport& operator=(const minlmreport &rhs);
    virtual ~minlmreport();
    ae_int_t &iterationscount;
    ae_int_t &terminationtype;
    ae_int_t &nfunc;
    ae_int_t &njac;
    ae_int_t &ngrad;
    ae_int_t &nhess;
    ae_int_t &ncholesky;

};
#endif

#if defined(AE_COMPILE_CQMODELS) || !defined(AE_PARTIAL_BUILD)

#endif

#if defined(AE_COMPILE_OPTGUARDAPI) || !defined(AE_PARTIAL_BUILD)

#endif

#if defined(AE_COMPILE_OPTSERV) || !defined(AE_PARTIAL_BUILD)

#endif

#if defined(AE_COMPILE_SNNLS) || !defined(AE_PARTIAL_BUILD)

#endif

#if defined(AE_COMPILE_SACTIVESETS) || !defined(AE_PARTIAL_BUILD)

#endif

#if defined(AE_COMPILE_QQPSOLVER) || !defined(AE_PARTIAL_BUILD)

#endif

#if defined(AE_COMPILE_MINLBFGS) || !defined(AE_PARTIAL_BUILD)
/*************************************************************************
        LIMITED MEMORY BFGS METHOD FOR LARGE SCALE OPTIMIZATION

DESCRIPTION:
The subroutine minimizes function F(x) of N arguments by  using  a  quasi-
Newton method (LBFGS scheme) which is optimized to use  a  minimum  amount
of memory.
The subroutine generates the approximation of an inverse Hessian matrix by
using information about the last M steps of the algorithm  (instead of N).
It lessens a required amount of memory from a value  of  order  N^2  to  a
value of order 2*N*M.


REQUIREMENTS:
Algorithm will request following information during its operation:
* function value F and its gradient G (simultaneously) at given point X


USAGE:
1. User initializes algorithm state with MinLBFGSCreate() call
2. User tunes solver parameters with MinLBFGSSetCond() MinLBFGSSetStpMax()
   and other functions
3. User calls MinLBFGSOptimize() function which takes algorithm  state and
   pointer (delegate, etc.) to callback function which calculates F/G.
4. User calls MinLBFGSResults() to get solution
5. Optionally user may call MinLBFGSRestartFrom() to solve another problem
   with same N/M but another starting point and/or another function.
   MinLBFGSRestartFrom() allows to reuse already initialized structure.


INPUT PARAMETERS:
    N       -   problem dimension. N>0
    M       -   number of corrections in the BFGS scheme of Hessian
                approximation update. Recommended value:  3<=M<=7. The smaller
                value causes worse convergence, the bigger will  not  cause  a
                considerably better convergence, but will cause a fall in  the
                performance. M<=N.
    X       -   initial solution approximation, array[0..N-1].


OUTPUT PARAMETERS:
    State   -   structure which stores algorithm state


NOTES:
1. you may tune stopping conditions with MinLBFGSSetCond() function
2. if target function contains exp() or other fast growing functions,  and
   optimization algorithm makes too large steps which leads  to  overflow,
   use MinLBFGSSetStpMax() function to bound algorithm's  steps.  However,
   L-BFGS rarely needs such a tuning.


  -- ALGLIB --
     Copyright 02.04.2010 by Bochkanov Sergey
*************************************************************************/
void minlbfgscreate(const ae_int_t n, const ae_int_t m, const real_1d_array &x, minlbfgsstate &state, const xparams _xparams = alglib::xdefault);
void minlbfgscreate(const ae_int_t m, const real_1d_array &x, minlbfgsstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
The subroutine is finite difference variant of MinLBFGSCreate().  It  uses
finite differences in order to differentiate target function.

Description below contains information which is specific to  this function
only. We recommend to read comments on MinLBFGSCreate() in  order  to  get
more information about creation of LBFGS optimizer.

INPUT PARAMETERS:
    N       -   problem dimension, N>0:
                * if given, only leading N elements of X are used
                * if not given, automatically determined from size of X
    M       -   number of corrections in the BFGS scheme of Hessian
                approximation update. Recommended value:  3<=M<=7. The smaller
                value causes worse convergence, the bigger will  not  cause  a
                considerably better convergence, but will cause a fall in  the
                performance. M<=N.
    X       -   starting point, array[0..N-1].
    DiffStep-   differentiation step, >0

OUTPUT PARAMETERS:
    State   -   structure which stores algorithm state

NOTES:
1. algorithm uses 4-point central formula for differentiation.
2. differentiation step along I-th axis is equal to DiffStep*S[I] where
   S[] is scaling vector which can be set by MinLBFGSSetScale() call.
3. we recommend you to use moderate values of  differentiation  step.  Too
   large step will result in too large truncation  errors, while too small
   step will result in too large numerical  errors.  1.0E-6  can  be  good
   value to start with.
4. Numerical  differentiation  is   very   inefficient  -   one   gradient
   calculation needs 4*N function evaluations. This function will work for
   any N - either small (1...10), moderate (10...100) or  large  (100...).
   However, performance penalty will be too severe for any N's except  for
   small ones.
   We should also say that code which relies on numerical  differentiation
   is   less  robust  and  precise.  LBFGS  needs  exact  gradient values.
   Imprecise gradient may slow  down  convergence,  especially  on  highly
   nonlinear problems.
   Thus  we  recommend to use this function for fast prototyping on small-
   dimensional problems only, and to implement analytical gradient as soon
   as possible.

  -- ALGLIB --
     Copyright 16.05.2011 by Bochkanov Sergey
*************************************************************************/
void minlbfgscreatef(const ae_int_t n, const ae_int_t m, const real_1d_array &x, const double diffstep, minlbfgsstate &state, const xparams _xparams = alglib::xdefault);
void minlbfgscreatef(const ae_int_t m, const real_1d_array &x, const double diffstep, minlbfgsstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets stopping conditions for L-BFGS optimization algorithm.

INPUT PARAMETERS:
    State   -   structure which stores algorithm state
    EpsG    -   >=0
                The  subroutine  finishes  its  work   if   the  condition
                |v|<EpsG is satisfied, where:
                * |.| means Euclidian norm
                * v - scaled gradient vector, v[i]=g[i]*s[i]
                * g - gradient
                * s - scaling coefficients set by MinLBFGSSetScale()
    EpsF    -   >=0
                The  subroutine  finishes  its work if on k+1-th iteration
                the  condition  |F(k+1)-F(k)|<=EpsF*max{|F(k)|,|F(k+1)|,1}
                is satisfied.
    EpsX    -   >=0
                The subroutine finishes its work if  on  k+1-th  iteration
                the condition |v|<=EpsX is fulfilled, where:
                * |.| means Euclidian norm
                * v - scaled step vector, v[i]=dx[i]/s[i]
                * dx - ste pvector, dx=X(k+1)-X(k)
                * s - scaling coefficients set by MinLBFGSSetScale()
    MaxIts  -   maximum number of iterations. If MaxIts=0, the  number  of
                iterations is unlimited.

Passing EpsG=0, EpsF=0, EpsX=0 and MaxIts=0 (simultaneously) will lead to
automatic stopping criterion selection (small EpsX).

  -- ALGLIB --
     Copyright 02.04.2010 by Bochkanov Sergey
*************************************************************************/
void minlbfgssetcond(const minlbfgsstate &state, const double epsg, const double epsf, const double epsx, const ae_int_t maxits, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function turns on/off reporting.

INPUT PARAMETERS:
    State   -   structure which stores algorithm state
    NeedXRep-   whether iteration reports are needed or not

If NeedXRep is True, algorithm will call rep() callback function if  it is
provided to MinLBFGSOptimize().


  -- ALGLIB --
     Copyright 02.04.2010 by Bochkanov Sergey
*************************************************************************/
void minlbfgssetxrep(const minlbfgsstate &state, const bool needxrep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets maximum step length

INPUT PARAMETERS:
    State   -   structure which stores algorithm state
    StpMax  -   maximum step length, >=0. Set StpMax to 0.0 (default),  if
                you don't want to limit step length.

Use this subroutine when you optimize target function which contains exp()
or  other  fast  growing  functions,  and optimization algorithm makes too
large  steps  which  leads  to overflow. This function allows us to reject
steps  that  are  too  large  (and  therefore  expose  us  to the possible
overflow) without actually calculating function value at the x+stp*d.

  -- ALGLIB --
     Copyright 02.04.2010 by Bochkanov Sergey
*************************************************************************/
void minlbfgssetstpmax(const minlbfgsstate &state, const double stpmax, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets scaling coefficients for LBFGS optimizer.

ALGLIB optimizers use scaling matrices to test stopping  conditions  (step
size and gradient are scaled before comparison with tolerances).  Scale of
the I-th variable is a translation invariant measure of:
a) "how large" the variable is
b) how large the step should be to make significant changes in the function

Scaling is also used by finite difference variant of the optimizer  - step
along I-th axis is equal to DiffStep*S[I].

In  most  optimizers  (and  in  the  LBFGS  too)  scaling is NOT a form of
preconditioning. It just  affects  stopping  conditions.  You  should  set
preconditioner  by  separate  call  to  one  of  the  MinLBFGSSetPrec...()
functions.

There  is  special  preconditioning  mode, however,  which  uses   scaling
coefficients to form diagonal preconditioning matrix. You  can  turn  this
mode on, if you want.   But  you should understand that scaling is not the
same thing as preconditioning - these are two different, although  related
forms of tuning solver.

INPUT PARAMETERS:
    State   -   structure stores algorithm state
    S       -   array[N], non-zero scaling coefficients
                S[i] may be negative, sign doesn't matter.

  -- ALGLIB --
     Copyright 14.01.2011 by Bochkanov Sergey
*************************************************************************/
void minlbfgssetscale(const minlbfgsstate &state, const real_1d_array &s, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Modification  of  the  preconditioner:  default  preconditioner    (simple
scaling, same for all elements of X) is used.

INPUT PARAMETERS:
    State   -   structure which stores algorithm state

NOTE:  you  can  change  preconditioner  "on  the  fly",  during algorithm
iterations.

  -- ALGLIB --
     Copyright 13.10.2010 by Bochkanov Sergey
*************************************************************************/
void minlbfgssetprecdefault(const minlbfgsstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Modification of the preconditioner: Cholesky factorization of  approximate
Hessian is used.

INPUT PARAMETERS:
    State   -   structure which stores algorithm state
    P       -   triangular preconditioner, Cholesky factorization of
                the approximate Hessian. array[0..N-1,0..N-1],
                (if larger, only leading N elements are used).
    IsUpper -   whether upper or lower triangle of P is given
                (other triangle is not referenced)

After call to this function preconditioner is changed to P  (P  is  copied
into the internal buffer).

NOTE:  you  can  change  preconditioner  "on  the  fly",  during algorithm
iterations.

NOTE 2:  P  should  be nonsingular. Exception will be thrown otherwise.

  -- ALGLIB --
     Copyright 13.10.2010 by Bochkanov Sergey
*************************************************************************/
void minlbfgssetpreccholesky(const minlbfgsstate &state, const real_2d_array &p, const bool isupper, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Modification  of  the  preconditioner:  diagonal of approximate Hessian is
used.

INPUT PARAMETERS:
    State   -   structure which stores algorithm state
    D       -   diagonal of the approximate Hessian, array[0..N-1],
                (if larger, only leading N elements are used).

NOTE:  you  can  change  preconditioner  "on  the  fly",  during algorithm
iterations.

NOTE 2: D[i] should be positive. Exception will be thrown otherwise.

NOTE 3: you should pass diagonal of approximate Hessian - NOT ITS INVERSE.

  -- ALGLIB --
     Copyright 13.10.2010 by Bochkanov Sergey
*************************************************************************/
void minlbfgssetprecdiag(const minlbfgsstate &state, const real_1d_array &d, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Modification of the preconditioner: scale-based diagonal preconditioning.

This preconditioning mode can be useful when you  don't  have  approximate
diagonal of Hessian, but you know that your  variables  are  badly  scaled
(for  example,  one  variable is in [1,10], and another in [1000,100000]),
and most part of the ill-conditioning comes from different scales of vars.

In this case simple  scale-based  preconditioner,  with H[i] = 1/(s[i]^2),
can greatly improve convergence.

IMPRTANT: you should set scale of your variables  with  MinLBFGSSetScale()
call  (before  or after MinLBFGSSetPrecScale() call). Without knowledge of
the scale of your variables scale-based preconditioner will be  just  unit
matrix.

INPUT PARAMETERS:
    State   -   structure which stores algorithm state

  -- ALGLIB --
     Copyright 13.10.2010 by Bochkanov Sergey
*************************************************************************/
void minlbfgssetprecscale(const minlbfgsstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function provides reverse communication interface
Reverse communication interface is not documented or recommended to use.
See below for functions which provide better documented API
*************************************************************************/
bool minlbfgsiteration(const minlbfgsstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This family of functions is used to launcn iterations of nonlinear optimizer

These functions accept following parameters:
    state   -   algorithm state
    func    -   callback which calculates function (or merit function)
                value func at given point x
    grad    -   callback which calculates function (or merit function)
                value func and gradient grad at given point x
    rep     -   optional callback which is called after each iteration
                can be NULL
    ptr     -   optional pointer which is passed to func/grad/hess/jac/rep
                can be NULL

NOTES:

1. This function has two different implementations: one which  uses  exact
   (analytical) user-supplied gradient,  and one which uses function value
   only  and  numerically  differentiates  function  in  order  to  obtain
   gradient.

   Depending  on  the  specific  function  used to create optimizer object
   (either MinLBFGSCreate() for analytical gradient  or  MinLBFGSCreateF()
   for numerical differentiation) you should choose appropriate variant of
   MinLBFGSOptimize() - one  which  accepts  function  AND gradient or one
   which accepts function ONLY.

   Be careful to choose variant of MinLBFGSOptimize() which corresponds to
   your optimization scheme! Table below lists different  combinations  of
   callback (function/gradient) passed to MinLBFGSOptimize()  and specific
   function used to create optimizer.


                     |         USER PASSED TO MinLBFGSOptimize()
   CREATED WITH      |  function only   |  function and gradient
   ------------------------------------------------------------
   MinLBFGSCreateF() |     work                FAIL
   MinLBFGSCreate()  |     FAIL                work

   Here "FAIL" denotes inappropriate combinations  of  optimizer  creation
   function  and  MinLBFGSOptimize()  version.   Attemps   to   use   such
   combination (for example, to create optimizer with MinLBFGSCreateF() and
   to pass gradient information to MinCGOptimize()) will lead to exception
   being thrown. Either  you  did  not pass gradient when it WAS needed or
   you passed gradient when it was NOT needed.

  -- ALGLIB --
     Copyright 20.03.2009 by Bochkanov Sergey

*************************************************************************/
void minlbfgsoptimize(minlbfgsstate &state,
    void (*func)(const real_1d_array &x, double &func, void *ptr),
    void  (*rep)(const real_1d_array &x, double func, void *ptr) = NULL,
    void *ptr = NULL,
    const xparams _xparams = alglib::xdefault);
void minlbfgsoptimize(minlbfgsstate &state,
    void (*grad)(const real_1d_array &x, double &func, real_1d_array &grad, void *ptr),
    void  (*rep)(const real_1d_array &x, double func, void *ptr) = NULL,
    void *ptr = NULL,
    const xparams _xparams = alglib::xdefault);


/*************************************************************************
This  function  activates/deactivates verification  of  the  user-supplied
analytic gradient.

Upon  activation  of  this  option  OptGuard  integrity  checker  performs
numerical differentiation of your target function  at  the  initial  point
(note: future versions may also perform check  at  the  final  point)  and
compares numerical gradient with analytic one provided by you.

If difference is too large, an error flag is set and optimization  session
continues. After optimization session is over, you can retrieve the report
which  stores  both  gradients  and  specific  components  highlighted  as
suspicious by the OptGuard.

The primary OptGuard report can be retrieved with minlbfgsoptguardresults().

IMPORTANT: gradient check is a high-overhead option which  will  cost  you
           about 3*N additional function evaluations. In many cases it may
           cost as much as the rest of the optimization session.

           YOU SHOULD NOT USE IT IN THE PRODUCTION CODE UNLESS YOU WANT TO
           CHECK DERIVATIVES PROVIDED BY SOME THIRD PARTY.

NOTE: unlike previous incarnation of the gradient checking code,  OptGuard
      does NOT interrupt optimization even if it discovers bad gradient.

INPUT PARAMETERS:
    State       -   structure used to store algorithm state
    TestStep    -   verification step used for numerical differentiation:
                    * TestStep=0 turns verification off
                    * TestStep>0 activates verification
                    You should carefully choose TestStep. Value  which  is
                    too large (so large that  function  behavior  is  non-
                    cubic at this scale) will lead  to  false  alarms. Too
                    short step will result in rounding  errors  dominating
                    numerical derivative.

                    You may use different step for different parameters by
                    means of setting scale with minlbfgssetscale().

=== EXPLANATION ==========================================================

In order to verify gradient algorithm performs following steps:
  * two trial steps are made to X[i]-TestStep*S[i] and X[i]+TestStep*S[i],
    where X[i] is i-th component of the initial point and S[i] is a  scale
    of i-th parameter
  * F(X) is evaluated at these trial points
  * we perform one more evaluation in the middle point of the interval
  * we  build  cubic  model using function values and derivatives at trial
    points and we compare its prediction with actual value in  the  middle
    point

  -- ALGLIB --
     Copyright 15.06.2014 by Bochkanov Sergey
*************************************************************************/
void minlbfgsoptguardgradient(const minlbfgsstate &state, const double teststep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This  function  activates/deactivates nonsmoothness monitoring  option  of
the  OptGuard  integrity  checker. Smoothness  monitor  silently  observes
solution process and tries to detect ill-posed problems, i.e. ones with:
a) discontinuous target function (non-C0)
b) nonsmooth     target function (non-C1)

Smoothness monitoring does NOT interrupt optimization  even if it suspects
that your problem is nonsmooth. It just sets corresponding  flags  in  the
OptGuard report which can be retrieved after optimization is over.

Smoothness monitoring is a moderate overhead option which often adds  less
than 1% to the optimizer running time. Thus, you can use it even for large
scale problems.

NOTE: OptGuard does  NOT  guarantee  that  it  will  always  detect  C0/C1
      continuity violations.

      First, minor errors are hard to  catch - say, a 0.0001 difference in
      the model values at two sides of the gap may be due to discontinuity
      of the model - or simply because the model has changed.

      Second, C1-violations  are  especially  difficult  to  detect  in  a
      noninvasive way. The optimizer usually  performs  very  short  steps
      near the nonsmoothness, and differentiation  usually   introduces  a
      lot of numerical noise.  It  is  hard  to  tell  whether  some  tiny
      discontinuity in the slope is due to real nonsmoothness or just  due
      to numerical noise alone.

      Our top priority was to avoid false positives, so in some rare cases
      minor errors may went unnoticed (however, in most cases they can  be
      spotted with restart from different initial point).

INPUT PARAMETERS:
    state   -   algorithm state
    level   -   monitoring level:
                * 0 - monitoring is disabled
                * 1 - noninvasive low-overhead monitoring; function values
                      and/or gradients are recorded, but OptGuard does not
                      try to perform additional evaluations  in  order  to
                      get more information about suspicious locations.

=== EXPLANATION ==========================================================

One major source of headache during optimization  is  the  possibility  of
the coding errors in the target function/constraints (or their gradients).
Such  errors   most   often   manifest   themselves  as  discontinuity  or
nonsmoothness of the target/constraints.

Another frequent situation is when you try to optimize something involving
lots of min() and max() operations, i.e. nonsmooth target. Although not  a
coding error, it is nonsmoothness anyway - and smooth  optimizers  usually
stop right after encountering nonsmoothness, well before reaching solution.

OptGuard integrity checker helps you to catch such situations: it monitors
function values/gradients being passed  to  the  optimizer  and  tries  to
errors. Upon discovering suspicious pair of points it  raises  appropriate
flag (and allows you to continue optimization). When optimization is done,
you can study OptGuard result.

  -- ALGLIB --
     Copyright 21.11.2018 by Bochkanov Sergey
*************************************************************************/
void minlbfgsoptguardsmoothness(const minlbfgsstate &state, const ae_int_t level, const xparams _xparams = alglib::xdefault);
void minlbfgsoptguardsmoothness(const minlbfgsstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Results of OptGuard integrity check, should be called  after  optimization
session is over.

=== PRIMARY REPORT =======================================================

OptGuard performs several checks which are intended to catch common errors
in the implementation of nonlinear function/gradient:
* incorrect analytic gradient
* discontinuous (non-C0) target functions (constraints)
* nonsmooth     (non-C1) target functions (constraints)

Each of these checks is activated with appropriate function:
* minlbfgsoptguardgradient() for gradient verification
* minlbfgsoptguardsmoothness() for C0/C1 checks

Following flags are set when these errors are suspected:
* rep.badgradsuspected, and additionally:
  * rep.badgradvidx for specific variable (gradient element) suspected
  * rep.badgradxbase, a point where gradient is tested
  * rep.badgraduser, user-provided gradient  (stored  as  2D  matrix  with
    single row in order to make  report  structure  compatible  with  more
    complex optimizers like MinNLC or MinLM)
  * rep.badgradnum,   reference    gradient    obtained    via   numerical
    differentiation (stored as  2D matrix with single row in order to make
    report structure compatible with more complex optimizers  like  MinNLC
    or MinLM)
* rep.nonc0suspected
* rep.nonc1suspected

=== ADDITIONAL REPORTS/LOGS ==============================================

Several different tests are performed to catch C0/C1 errors, you can  find
out specific test signaled error by looking to:
* rep.nonc0test0positive, for non-C0 test #0
* rep.nonc1test0positive, for non-C1 test #0
* rep.nonc1test1positive, for non-C1 test #1

Additional information (including line search logs)  can  be  obtained  by
means of:
* minlbfgsoptguardnonc1test0results()
* minlbfgsoptguardnonc1test1results()
which return detailed error reports, specific points where discontinuities
were found, and so on.

==========================================================================

INPUT PARAMETERS:
    state   -   algorithm state

OUTPUT PARAMETERS:
    rep     -   generic OptGuard report;  more  detailed  reports  can  be
                retrieved with other functions.

NOTE: false negatives (nonsmooth problems are not identified as  nonsmooth
      ones) are possible although unlikely.

      The reason  is  that  you  need  to  make several evaluations around
      nonsmoothness  in  order  to  accumulate  enough  information  about
      function curvature. Say, if you start right from the nonsmooth point,
      optimizer simply won't get enough data to understand what  is  going
      wrong before it terminates due to abrupt changes in the  derivative.
      It is also  possible  that  "unlucky"  step  will  move  us  to  the
      termination too quickly.

      Our current approach is to have less than 0.1%  false  negatives  in
      our test examples  (measured  with  multiple  restarts  from  random
      points), and to have exactly 0% false positives.

  -- ALGLIB --
     Copyright 21.11.2018 by Bochkanov Sergey
*************************************************************************/
void minlbfgsoptguardresults(const minlbfgsstate &state, optguardreport &rep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Detailed results of the OptGuard integrity check for nonsmoothness test #0

Nonsmoothness (non-C1) test #0 studies  function  values  (not  gradient!)
obtained during line searches and monitors  behavior  of  the  directional
derivative estimate.

This test is less powerful than test #1, but it does  not  depend  on  the
gradient values and thus it is more robust against artifacts introduced by
numerical differentiation.

Two reports are returned:
* a "strongest" one, corresponding  to  line   search  which  had  highest
  value of the nonsmoothness indicator
* a "longest" one, corresponding to line search which  had  more  function
  evaluations, and thus is more detailed

In both cases following fields are returned:

* positive - is TRUE  when test flagged suspicious point;  FALSE  if  test
  did not notice anything (in the latter cases fields below are empty).
* x0[], d[] - arrays of length N which store initial point  and  direction
  for line search (d[] can be normalized, but does not have to)
* stp[], f[] - arrays of length CNT which store step lengths and  function
  values at these points; f[i] is evaluated in x0+stp[i]*d.
* stpidxa, stpidxb - we  suspect  that  function  violates  C1  continuity
  between steps #stpidxa and #stpidxb (usually we have  stpidxb=stpidxa+3,
  with  most  likely  position  of  the  violation  between  stpidxa+1 and
  stpidxa+2.

==========================================================================
= SHORTLY SPEAKING: build a 2D plot of (stp,f) and look at it -  you  will
=                   see where C1 continuity is violated.
==========================================================================

INPUT PARAMETERS:
    state   -   algorithm state

OUTPUT PARAMETERS:
    strrep  -   C1 test #0 "strong" report
    lngrep  -   C1 test #0 "long" report

  -- ALGLIB --
     Copyright 21.11.2018 by Bochkanov Sergey
*************************************************************************/
void minlbfgsoptguardnonc1test0results(const minlbfgsstate &state, optguardnonc1test0report &strrep, optguardnonc1test0report &lngrep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Detailed results of the OptGuard integrity check for nonsmoothness test #1

Nonsmoothness (non-C1)  test  #1  studies  individual  components  of  the
gradient computed during line search.

When precise analytic gradient is provided this test is more powerful than
test #0  which  works  with  function  values  and  ignores  user-provided
gradient.  However,  test  #0  becomes  more   powerful   when   numerical
differentiation is employed (in such cases test #1 detects  higher  levels
of numerical noise and becomes too conservative).

This test also tells specific components of the gradient which violate  C1
continuity, which makes it more informative than #0, which just tells that
continuity is violated.

Two reports are returned:
* a "strongest" one, corresponding  to  line   search  which  had  highest
  value of the nonsmoothness indicator
* a "longest" one, corresponding to line search which  had  more  function
  evaluations, and thus is more detailed

In both cases following fields are returned:

* positive - is TRUE  when test flagged suspicious point;  FALSE  if  test
  did not notice anything (in the latter cases fields below are empty).
* vidx - is an index of the variable in [0,N) with nonsmooth derivative
* x0[], d[] - arrays of length N which store initial point  and  direction
  for line search (d[] can be normalized, but does not have to)
* stp[], g[] - arrays of length CNT which store step lengths and  gradient
  values at these points; g[i] is evaluated in  x0+stp[i]*d  and  contains
  vidx-th component of the gradient.
* stpidxa, stpidxb - we  suspect  that  function  violates  C1  continuity
  between steps #stpidxa and #stpidxb (usually we have  stpidxb=stpidxa+3,
  with  most  likely  position  of  the  violation  between  stpidxa+1 and
  stpidxa+2.

==========================================================================
= SHORTLY SPEAKING: build a 2D plot of (stp,f) and look at it -  you  will
=                   see where C1 continuity is violated.
==========================================================================

INPUT PARAMETERS:
    state   -   algorithm state

OUTPUT PARAMETERS:
    strrep  -   C1 test #1 "strong" report
    lngrep  -   C1 test #1 "long" report

  -- ALGLIB --
     Copyright 21.11.2018 by Bochkanov Sergey
*************************************************************************/
void minlbfgsoptguardnonc1test1results(const minlbfgsstate &state, optguardnonc1test1report &strrep, optguardnonc1test1report &lngrep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
L-BFGS algorithm results

INPUT PARAMETERS:
    State   -   algorithm state

OUTPUT PARAMETERS:
    X       -   array[0..N-1], solution
    Rep     -   optimization report:
                * Rep.TerminationType completetion code:
                    * -8    internal integrity control  detected  infinite
                            or NAN values in  function/gradient.  Abnormal
                            termination signalled.
                    * -2    rounding errors prevent further improvement.
                            X contains best point found.
                    * -1    incorrect parameters were specified
                    *  1    relative function improvement is no more than
                            EpsF.
                    *  2    relative step is no more than EpsX.
                    *  4    gradient norm is no more than EpsG
                    *  5    MaxIts steps was taken
                    *  7    stopping conditions are too stringent,
                            further improvement is impossible
                    *  8    terminated by user who called minlbfgsrequesttermination().
                            X contains point which was "current accepted" when
                            termination request was submitted.
                * Rep.IterationsCount contains iterations count
                * NFEV countains number of function calculations

  -- ALGLIB --
     Copyright 02.04.2010 by Bochkanov Sergey
*************************************************************************/
void minlbfgsresults(const minlbfgsstate &state, real_1d_array &x, minlbfgsreport &rep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
L-BFGS algorithm results

Buffered implementation of MinLBFGSResults which uses pre-allocated buffer
to store X[]. If buffer size is  too  small,  it  resizes  buffer.  It  is
intended to be used in the inner cycles of performance critical algorithms
where array reallocation penalty is too large to be ignored.

  -- ALGLIB --
     Copyright 20.08.2010 by Bochkanov Sergey
*************************************************************************/
void minlbfgsresultsbuf(const minlbfgsstate &state, real_1d_array &x, minlbfgsreport &rep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This  subroutine restarts LBFGS algorithm from new point. All optimization
parameters are left unchanged.

This  function  allows  to  solve multiple  optimization  problems  (which
must have same number of dimensions) without object reallocation penalty.

INPUT PARAMETERS:
    State   -   structure used to store algorithm state
    X       -   new starting point.

  -- ALGLIB --
     Copyright 30.07.2010 by Bochkanov Sergey
*************************************************************************/
void minlbfgsrestartfrom(const minlbfgsstate &state, const real_1d_array &x, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This subroutine submits request for termination of running  optimizer.  It
should be called from user-supplied callback when user decides that it  is
time to "smoothly" terminate optimization process.  As  result,  optimizer
stops at point which was "current accepted" when termination  request  was
submitted and returns error code 8 (successful termination).

INPUT PARAMETERS:
    State   -   optimizer structure

NOTE: after  request  for  termination  optimizer  may   perform   several
      additional calls to user-supplied callbacks. It does  NOT  guarantee
      to stop immediately - it just guarantees that these additional calls
      will be discarded later.

NOTE: calling this function on optimizer which is NOT running will have no
      effect.

NOTE: multiple calls to this function are possible. First call is counted,
      subsequent calls are silently ignored.

  -- ALGLIB --
     Copyright 08.10.2014 by Bochkanov Sergey
*************************************************************************/
void minlbfgsrequesttermination(const minlbfgsstate &state, const xparams _xparams = alglib::xdefault);
#endif

#if defined(AE_COMPILE_QPDENSEAULSOLVER) || !defined(AE_PARTIAL_BUILD)

#endif

#if defined(AE_COMPILE_MINBLEIC) || !defined(AE_PARTIAL_BUILD)
/*************************************************************************
                     BOUND CONSTRAINED OPTIMIZATION
       WITH ADDITIONAL LINEAR EQUALITY AND INEQUALITY CONSTRAINTS

DESCRIPTION:
The  subroutine  minimizes  function   F(x)  of N arguments subject to any
combination of:
* bound constraints
* linear inequality constraints
* linear equality constraints

REQUIREMENTS:
* user must provide function value and gradient
* starting point X0 must be feasible or
  not too far away from the feasible set
* grad(f) must be Lipschitz continuous on a level set:
  L = { x : f(x)<=f(x0) }
* function must be defined everywhere on the feasible set F

USAGE:

Constrained optimization if far more complex than the unconstrained one.
Here we give very brief outline of the BLEIC optimizer. We strongly recommend
you to read examples in the ALGLIB Reference Manual and to read ALGLIB User Guide
on optimization, which is available at http://www.alglib.net/optimization/

1. User initializes algorithm state with MinBLEICCreate() call

2. USer adds boundary and/or linear constraints by calling
   MinBLEICSetBC() and MinBLEICSetLC() functions.

3. User sets stopping conditions with MinBLEICSetCond().

4. User calls MinBLEICOptimize() function which takes algorithm  state and
   pointer (delegate, etc.) to callback function which calculates F/G.

5. User calls MinBLEICResults() to get solution

6. Optionally user may call MinBLEICRestartFrom() to solve another problem
   with same N but another starting point.
   MinBLEICRestartFrom() allows to reuse already initialized structure.

NOTE: if you have box-only constraints (no  general  linear  constraints),
      then MinBC optimizer can be better option. It uses  special,  faster
      constraint activation method, which performs better on problems with
      multiple constraints active at the solution.

      On small-scale problems performance of MinBC is similar to  that  of
      MinBLEIC, but on large-scale ones (hundreds and thousands of  active
      constraints) it can be several times faster than MinBLEIC.

INPUT PARAMETERS:
    N       -   problem dimension, N>0:
                * if given, only leading N elements of X are used
                * if not given, automatically determined from size ofX
    X       -   starting point, array[N]:
                * it is better to set X to a feasible point
                * but X can be infeasible, in which case algorithm will try
                  to find feasible point first, using X as initial
                  approximation.

OUTPUT PARAMETERS:
    State   -   structure stores algorithm state

  -- ALGLIB --
     Copyright 28.11.2010 by Bochkanov Sergey
*************************************************************************/
void minbleiccreate(const ae_int_t n, const real_1d_array &x, minbleicstate &state, const xparams _xparams = alglib::xdefault);
void minbleiccreate(const real_1d_array &x, minbleicstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
The subroutine is finite difference variant of MinBLEICCreate().  It  uses
finite differences in order to differentiate target function.

Description below contains information which is specific to  this function
only. We recommend to read comments on MinBLEICCreate() in  order  to  get
more information about creation of BLEIC optimizer.

INPUT PARAMETERS:
    N       -   problem dimension, N>0:
                * if given, only leading N elements of X are used
                * if not given, automatically determined from size of X
    X       -   starting point, array[0..N-1].
    DiffStep-   differentiation step, >0

OUTPUT PARAMETERS:
    State   -   structure which stores algorithm state

NOTES:
1. algorithm uses 4-point central formula for differentiation.
2. differentiation step along I-th axis is equal to DiffStep*S[I] where
   S[] is scaling vector which can be set by MinBLEICSetScale() call.
3. we recommend you to use moderate values of  differentiation  step.  Too
   large step will result in too large truncation  errors, while too small
   step will result in too large numerical  errors.  1.0E-6  can  be  good
   value to start with.
4. Numerical  differentiation  is   very   inefficient  -   one   gradient
   calculation needs 4*N function evaluations. This function will work for
   any N - either small (1...10), moderate (10...100) or  large  (100...).
   However, performance penalty will be too severe for any N's except  for
   small ones.
   We should also say that code which relies on numerical  differentiation
   is  less  robust and precise. CG needs exact gradient values. Imprecise
   gradient may slow  down  convergence, especially  on  highly  nonlinear
   problems.
   Thus  we  recommend to use this function for fast prototyping on small-
   dimensional problems only, and to implement analytical gradient as soon
   as possible.

  -- ALGLIB --
     Copyright 16.05.2011 by Bochkanov Sergey
*************************************************************************/
void minbleiccreatef(const ae_int_t n, const real_1d_array &x, const double diffstep, minbleicstate &state, const xparams _xparams = alglib::xdefault);
void minbleiccreatef(const real_1d_array &x, const double diffstep, minbleicstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets boundary constraints for BLEIC optimizer.

Boundary constraints are inactive by default (after initial creation).
They are preserved after algorithm restart with MinBLEICRestartFrom().

NOTE: if you have box-only constraints (no  general  linear  constraints),
      then MinBC optimizer can be better option. It uses  special,  faster
      constraint activation method, which performs better on problems with
      multiple constraints active at the solution.

      On small-scale problems performance of MinBC is similar to  that  of
      MinBLEIC, but on large-scale ones (hundreds and thousands of  active
      constraints) it can be several times faster than MinBLEIC.

INPUT PARAMETERS:
    State   -   structure stores algorithm state
    BndL    -   lower bounds, array[N].
                If some (all) variables are unbounded, you may specify
                very small number or -INF.
    BndU    -   upper bounds, array[N].
                If some (all) variables are unbounded, you may specify
                very large number or +INF.

NOTE 1: it is possible to specify BndL[i]=BndU[i]. In this case I-th
variable will be "frozen" at X[i]=BndL[i]=BndU[i].

NOTE 2: this solver has following useful properties:
* bound constraints are always satisfied exactly
* function is evaluated only INSIDE area specified by  bound  constraints,
  even  when  numerical  differentiation is used (algorithm adjusts  nodes
  according to boundary constraints)

  -- ALGLIB --
     Copyright 28.11.2010 by Bochkanov Sergey
*************************************************************************/
void minbleicsetbc(const minbleicstate &state, const real_1d_array &bndl, const real_1d_array &bndu, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets linear constraints for BLEIC optimizer.

Linear constraints are inactive by default (after initial creation).
They are preserved after algorithm restart with MinBLEICRestartFrom().

INPUT PARAMETERS:
    State   -   structure previously allocated with MinBLEICCreate call.
    C       -   linear constraints, array[K,N+1].
                Each row of C represents one constraint, either equality
                or inequality (see below):
                * first N elements correspond to coefficients,
                * last element corresponds to the right part.
                All elements of C (including right part) must be finite.
    CT      -   type of constraints, array[K]:
                * if CT[i]>0, then I-th constraint is C[i,*]*x >= C[i,n]
                * if CT[i]=0, then I-th constraint is C[i,*]*x  = C[i,n]
                * if CT[i]<0, then I-th constraint is C[i,*]*x <= C[i,n]
    K       -   number of equality/inequality constraints, K>=0:
                * if given, only leading K elements of C/CT are used
                * if not given, automatically determined from sizes of C/CT

NOTE 1: linear (non-bound) constraints are satisfied only approximately:
* there always exists some minor violation (about Epsilon in magnitude)
  due to rounding errors
* numerical differentiation, if used, may  lead  to  function  evaluations
  outside  of the feasible  area,   because   algorithm  does  NOT  change
  numerical differentiation formula according to linear constraints.
If you want constraints to be  satisfied  exactly, try to reformulate your
problem  in  such  manner  that  all constraints will become boundary ones
(this kind of constraints is always satisfied exactly, both in  the  final
solution and in all intermediate points).

  -- ALGLIB --
     Copyright 28.11.2010 by Bochkanov Sergey
*************************************************************************/
void minbleicsetlc(const minbleicstate &state, const real_2d_array &c, const integer_1d_array &ct, const ae_int_t k, const xparams _xparams = alglib::xdefault);
void minbleicsetlc(const minbleicstate &state, const real_2d_array &c, const integer_1d_array &ct, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets stopping conditions for the optimizer.

INPUT PARAMETERS:
    State   -   structure which stores algorithm state
    EpsG    -   >=0
                The  subroutine  finishes  its  work   if   the  condition
                |v|<EpsG is satisfied, where:
                * |.| means Euclidian norm
                * v - scaled gradient vector, v[i]=g[i]*s[i]
                * g - gradient
                * s - scaling coefficients set by MinBLEICSetScale()
    EpsF    -   >=0
                The  subroutine  finishes  its work if on k+1-th iteration
                the  condition  |F(k+1)-F(k)|<=EpsF*max{|F(k)|,|F(k+1)|,1}
                is satisfied.
    EpsX    -   >=0
                The subroutine finishes its work if  on  k+1-th  iteration
                the condition |v|<=EpsX is fulfilled, where:
                * |.| means Euclidian norm
                * v - scaled step vector, v[i]=dx[i]/s[i]
                * dx - step vector, dx=X(k+1)-X(k)
                * s - scaling coefficients set by MinBLEICSetScale()
    MaxIts  -   maximum number of iterations. If MaxIts=0, the  number  of
                iterations is unlimited.

Passing EpsG=0, EpsF=0 and EpsX=0 and MaxIts=0 (simultaneously) will lead
to automatic stopping criterion selection.

NOTE: when SetCond() called with non-zero MaxIts, BLEIC solver may perform
      slightly more than MaxIts iterations. I.e., MaxIts  sets  non-strict
      limit on iterations count.

  -- ALGLIB --
     Copyright 28.11.2010 by Bochkanov Sergey
*************************************************************************/
void minbleicsetcond(const minbleicstate &state, const double epsg, const double epsf, const double epsx, const ae_int_t maxits, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets scaling coefficients for BLEIC optimizer.

ALGLIB optimizers use scaling matrices to test stopping  conditions  (step
size and gradient are scaled before comparison with tolerances).  Scale of
the I-th variable is a translation invariant measure of:
a) "how large" the variable is
b) how large the step should be to make significant changes in the function

Scaling is also used by finite difference variant of the optimizer  - step
along I-th axis is equal to DiffStep*S[I].

In  most  optimizers  (and  in  the  BLEIC  too)  scaling is NOT a form of
preconditioning. It just  affects  stopping  conditions.  You  should  set
preconditioner  by  separate  call  to  one  of  the  MinBLEICSetPrec...()
functions.

There is a special  preconditioning  mode, however,  which  uses   scaling
coefficients to form diagonal preconditioning matrix. You  can  turn  this
mode on, if you want.   But  you should understand that scaling is not the
same thing as preconditioning - these are two different, although  related
forms of tuning solver.

INPUT PARAMETERS:
    State   -   structure stores algorithm state
    S       -   array[N], non-zero scaling coefficients
                S[i] may be negative, sign doesn't matter.

  -- ALGLIB --
     Copyright 14.01.2011 by Bochkanov Sergey
*************************************************************************/
void minbleicsetscale(const minbleicstate &state, const real_1d_array &s, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Modification of the preconditioner: preconditioning is turned off.

INPUT PARAMETERS:
    State   -   structure which stores algorithm state

  -- ALGLIB --
     Copyright 13.10.2010 by Bochkanov Sergey
*************************************************************************/
void minbleicsetprecdefault(const minbleicstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Modification  of  the  preconditioner:  diagonal of approximate Hessian is
used.

INPUT PARAMETERS:
    State   -   structure which stores algorithm state
    D       -   diagonal of the approximate Hessian, array[0..N-1],
                (if larger, only leading N elements are used).

NOTE 1: D[i] should be positive. Exception will be thrown otherwise.

NOTE 2: you should pass diagonal of approximate Hessian - NOT ITS INVERSE.

  -- ALGLIB --
     Copyright 13.10.2010 by Bochkanov Sergey
*************************************************************************/
void minbleicsetprecdiag(const minbleicstate &state, const real_1d_array &d, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Modification of the preconditioner: scale-based diagonal preconditioning.

This preconditioning mode can be useful when you  don't  have  approximate
diagonal of Hessian, but you know that your  variables  are  badly  scaled
(for  example,  one  variable is in [1,10], and another in [1000,100000]),
and most part of the ill-conditioning comes from different scales of vars.

In this case simple  scale-based  preconditioner,  with H[i] = 1/(s[i]^2),
can greatly improve convergence.

IMPRTANT: you should set scale of your variables  with  MinBLEICSetScale()
call  (before  or after MinBLEICSetPrecScale() call). Without knowledge of
the scale of your variables scale-based preconditioner will be  just  unit
matrix.

INPUT PARAMETERS:
    State   -   structure which stores algorithm state

  -- ALGLIB --
     Copyright 13.10.2010 by Bochkanov Sergey
*************************************************************************/
void minbleicsetprecscale(const minbleicstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function turns on/off reporting.

INPUT PARAMETERS:
    State   -   structure which stores algorithm state
    NeedXRep-   whether iteration reports are needed or not

If NeedXRep is True, algorithm will call rep() callback function if  it is
provided to MinBLEICOptimize().

  -- ALGLIB --
     Copyright 28.11.2010 by Bochkanov Sergey
*************************************************************************/
void minbleicsetxrep(const minbleicstate &state, const bool needxrep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets maximum step length

IMPORTANT: this feature is hard to combine with preconditioning. You can't
set upper limit on step length, when you solve optimization  problem  with
linear (non-boundary) constraints AND preconditioner turned on.

When  non-boundary  constraints  are  present,  you  have to either a) use
preconditioner, or b) use upper limit on step length.  YOU CAN'T USE BOTH!
In this case algorithm will terminate with appropriate error code.

INPUT PARAMETERS:
    State   -   structure which stores algorithm state
    StpMax  -   maximum step length, >=0. Set StpMax to 0.0,  if you don't
                want to limit step length.

Use this subroutine when you optimize target function which contains exp()
or  other  fast  growing  functions,  and optimization algorithm makes too
large  steps  which  lead   to overflow. This function allows us to reject
steps  that  are  too  large  (and  therefore  expose  us  to the possible
overflow) without actually calculating function value at the x+stp*d.

  -- ALGLIB --
     Copyright 02.04.2010 by Bochkanov Sergey
*************************************************************************/
void minbleicsetstpmax(const minbleicstate &state, const double stpmax, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function provides reverse communication interface
Reverse communication interface is not documented or recommended to use.
See below for functions which provide better documented API
*************************************************************************/
bool minbleiciteration(const minbleicstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This family of functions is used to launcn iterations of nonlinear optimizer

These functions accept following parameters:
    state   -   algorithm state
    func    -   callback which calculates function (or merit function)
                value func at given point x
    grad    -   callback which calculates function (or merit function)
                value func and gradient grad at given point x
    rep     -   optional callback which is called after each iteration
                can be NULL
    ptr     -   optional pointer which is passed to func/grad/hess/jac/rep
                can be NULL

NOTES:

1. This function has two different implementations: one which  uses  exact
   (analytical) user-supplied gradient,  and one which uses function value
   only  and  numerically  differentiates  function  in  order  to  obtain
   gradient.

   Depending  on  the  specific  function  used to create optimizer object
   (either  MinBLEICCreate() for analytical gradient or  MinBLEICCreateF()
   for numerical differentiation) you should choose appropriate variant of
   MinBLEICOptimize() - one  which  accepts  function  AND gradient or one
   which accepts function ONLY.

   Be careful to choose variant of MinBLEICOptimize() which corresponds to
   your optimization scheme! Table below lists different  combinations  of
   callback (function/gradient) passed to MinBLEICOptimize()  and specific
   function used to create optimizer.


                     |         USER PASSED TO MinBLEICOptimize()
   CREATED WITH      |  function only   |  function and gradient
   ------------------------------------------------------------
   MinBLEICCreateF() |     work                FAIL
   MinBLEICCreate()  |     FAIL                work

   Here "FAIL" denotes inappropriate combinations  of  optimizer  creation
   function  and  MinBLEICOptimize()  version.   Attemps   to   use   such
   combination (for  example,  to  create optimizer with MinBLEICCreateF()
   and  to  pass  gradient information to MinBLEICOptimize()) will lead to
   exception being thrown. Either  you  did  not pass gradient when it WAS
   needed or you passed gradient when it was NOT needed.

  -- ALGLIB --
     Copyright 28.11.2010 by Bochkanov Sergey

*************************************************************************/
void minbleicoptimize(minbleicstate &state,
    void (*func)(const real_1d_array &x, double &func, void *ptr),
    void  (*rep)(const real_1d_array &x, double func, void *ptr) = NULL,
    void *ptr = NULL,
    const xparams _xparams = alglib::xdefault);
void minbleicoptimize(minbleicstate &state,
    void (*grad)(const real_1d_array &x, double &func, real_1d_array &grad, void *ptr),
    void  (*rep)(const real_1d_array &x, double func, void *ptr) = NULL,
    void *ptr = NULL,
    const xparams _xparams = alglib::xdefault);


/*************************************************************************
This  function  activates/deactivates verification  of  the  user-supplied
analytic gradient.

Upon  activation  of  this  option  OptGuard  integrity  checker  performs
numerical differentiation of your target function  at  the  initial  point
(note: future versions may also perform check  at  the  final  point)  and
compares numerical gradient with analytic one provided by you.

If difference is too large, an error flag is set and optimization  session
continues. After optimization session is over, you can retrieve the report
which  stores  both  gradients  and  specific  components  highlighted  as
suspicious by the OptGuard.

The primary OptGuard report can be retrieved with minbleicoptguardresults().

IMPORTANT: gradient check is a high-overhead option which  will  cost  you
           about 3*N additional function evaluations. In many cases it may
           cost as much as the rest of the optimization session.

           YOU SHOULD NOT USE IT IN THE PRODUCTION CODE UNLESS YOU WANT TO
           CHECK DERIVATIVES PROVIDED BY SOME THIRD PARTY.

NOTE: unlike previous incarnation of the gradient checking code,  OptGuard
      does NOT interrupt optimization even if it discovers bad gradient.

INPUT PARAMETERS:
    State       -   structure used to store algorithm state
    TestStep    -   verification step used for numerical differentiation:
                    * TestStep=0 turns verification off
                    * TestStep>0 activates verification
                    You should carefully choose TestStep. Value  which  is
                    too large (so large that  function  behavior  is  non-
                    cubic at this scale) will lead  to  false  alarms. Too
                    short step will result in rounding  errors  dominating
                    numerical derivative.

                    You may use different step for different parameters by
                    means of setting scale with minbleicsetscale().

=== EXPLANATION ==========================================================

In order to verify gradient algorithm performs following steps:
  * two trial steps are made to X[i]-TestStep*S[i] and X[i]+TestStep*S[i],
    where X[i] is i-th component of the initial point and S[i] is a  scale
    of i-th parameter
  * F(X) is evaluated at these trial points
  * we perform one more evaluation in the middle point of the interval
  * we  build  cubic  model using function values and derivatives at trial
    points and we compare its prediction with actual value in  the  middle
    point

  -- ALGLIB --
     Copyright 15.06.2014 by Bochkanov Sergey
*************************************************************************/
void minbleicoptguardgradient(const minbleicstate &state, const double teststep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This  function  activates/deactivates nonsmoothness monitoring  option  of
the  OptGuard  integrity  checker. Smoothness  monitor  silently  observes
solution process and tries to detect ill-posed problems, i.e. ones with:
a) discontinuous target function (non-C0)
b) nonsmooth     target function (non-C1)

Smoothness monitoring does NOT interrupt optimization  even if it suspects
that your problem is nonsmooth. It just sets corresponding  flags  in  the
OptGuard report which can be retrieved after optimization is over.

Smoothness monitoring is a moderate overhead option which often adds  less
than 1% to the optimizer running time. Thus, you can use it even for large
scale problems.

NOTE: OptGuard does  NOT  guarantee  that  it  will  always  detect  C0/C1
      continuity violations.

      First, minor errors are hard to  catch - say, a 0.0001 difference in
      the model values at two sides of the gap may be due to discontinuity
      of the model - or simply because the model has changed.

      Second, C1-violations  are  especially  difficult  to  detect  in  a
      noninvasive way. The optimizer usually  performs  very  short  steps
      near the nonsmoothness, and differentiation  usually   introduces  a
      lot of numerical noise.  It  is  hard  to  tell  whether  some  tiny
      discontinuity in the slope is due to real nonsmoothness or just  due
      to numerical noise alone.

      Our top priority was to avoid false positives, so in some rare cases
      minor errors may went unnoticed (however, in most cases they can  be
      spotted with restart from different initial point).

INPUT PARAMETERS:
    state   -   algorithm state
    level   -   monitoring level:
                * 0 - monitoring is disabled
                * 1 - noninvasive low-overhead monitoring; function values
                      and/or gradients are recorded, but OptGuard does not
                      try to perform additional evaluations  in  order  to
                      get more information about suspicious locations.

=== EXPLANATION ==========================================================

One major source of headache during optimization  is  the  possibility  of
the coding errors in the target function/constraints (or their gradients).
Such  errors   most   often   manifest   themselves  as  discontinuity  or
nonsmoothness of the target/constraints.

Another frequent situation is when you try to optimize something involving
lots of min() and max() operations, i.e. nonsmooth target. Although not  a
coding error, it is nonsmoothness anyway - and smooth  optimizers  usually
stop right after encountering nonsmoothness, well before reaching solution.

OptGuard integrity checker helps you to catch such situations: it monitors
function values/gradients being passed  to  the  optimizer  and  tries  to
errors. Upon discovering suspicious pair of points it  raises  appropriate
flag (and allows you to continue optimization). When optimization is done,
you can study OptGuard result.

  -- ALGLIB --
     Copyright 21.11.2018 by Bochkanov Sergey
*************************************************************************/
void minbleicoptguardsmoothness(const minbleicstate &state, const ae_int_t level, const xparams _xparams = alglib::xdefault);
void minbleicoptguardsmoothness(const minbleicstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Results of OptGuard integrity check, should be called  after  optimization
session is over.

=== PRIMARY REPORT =======================================================

OptGuard performs several checks which are intended to catch common errors
in the implementation of nonlinear function/gradient:
* incorrect analytic gradient
* discontinuous (non-C0) target functions (constraints)
* nonsmooth     (non-C1) target functions (constraints)

Each of these checks is activated with appropriate function:
* minbleicoptguardgradient() for gradient verification
* minbleicoptguardsmoothness() for C0/C1 checks

Following flags are set when these errors are suspected:
* rep.badgradsuspected, and additionally:
  * rep.badgradvidx for specific variable (gradient element) suspected
  * rep.badgradxbase, a point where gradient is tested
  * rep.badgraduser, user-provided gradient  (stored  as  2D  matrix  with
    single row in order to make  report  structure  compatible  with  more
    complex optimizers like MinNLC or MinLM)
  * rep.badgradnum,   reference    gradient    obtained    via   numerical
    differentiation (stored as  2D matrix with single row in order to make
    report structure compatible with more complex optimizers  like  MinNLC
    or MinLM)
* rep.nonc0suspected
* rep.nonc1suspected

=== ADDITIONAL REPORTS/LOGS ==============================================

Several different tests are performed to catch C0/C1 errors, you can  find
out specific test signaled error by looking to:
* rep.nonc0test0positive, for non-C0 test #0
* rep.nonc1test0positive, for non-C1 test #0
* rep.nonc1test1positive, for non-C1 test #1

Additional information (including line search logs)  can  be  obtained  by
means of:
* minbleicoptguardnonc1test0results()
* minbleicoptguardnonc1test1results()
which return detailed error reports, specific points where discontinuities
were found, and so on.

==========================================================================

INPUT PARAMETERS:
    state   -   algorithm state

OUTPUT PARAMETERS:
    rep     -   generic OptGuard report;  more  detailed  reports  can  be
                retrieved with other functions.

NOTE: false negatives (nonsmooth problems are not identified as  nonsmooth
      ones) are possible although unlikely.

      The reason  is  that  you  need  to  make several evaluations around
      nonsmoothness  in  order  to  accumulate  enough  information  about
      function curvature. Say, if you start right from the nonsmooth point,
      optimizer simply won't get enough data to understand what  is  going
      wrong before it terminates due to abrupt changes in the  derivative.
      It is also  possible  that  "unlucky"  step  will  move  us  to  the
      termination too quickly.

      Our current approach is to have less than 0.1%  false  negatives  in
      our test examples  (measured  with  multiple  restarts  from  random
      points), and to have exactly 0% false positives.

  -- ALGLIB --
     Copyright 21.11.2018 by Bochkanov Sergey
*************************************************************************/
void minbleicoptguardresults(const minbleicstate &state, optguardreport &rep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Detailed results of the OptGuard integrity check for nonsmoothness test #0

Nonsmoothness (non-C1) test #0 studies  function  values  (not  gradient!)
obtained during line searches and monitors  behavior  of  the  directional
derivative estimate.

This test is less powerful than test #1, but it does  not  depend  on  the
gradient values and thus it is more robust against artifacts introduced by
numerical differentiation.

Two reports are returned:
* a "strongest" one, corresponding  to  line   search  which  had  highest
  value of the nonsmoothness indicator
* a "longest" one, corresponding to line search which  had  more  function
  evaluations, and thus is more detailed

In both cases following fields are returned:

* positive - is TRUE  when test flagged suspicious point;  FALSE  if  test
  did not notice anything (in the latter cases fields below are empty).
* x0[], d[] - arrays of length N which store initial point  and  direction
  for line search (d[] can be normalized, but does not have to)
* stp[], f[] - arrays of length CNT which store step lengths and  function
  values at these points; f[i] is evaluated in x0+stp[i]*d.
* stpidxa, stpidxb - we  suspect  that  function  violates  C1  continuity
  between steps #stpidxa and #stpidxb (usually we have  stpidxb=stpidxa+3,
  with  most  likely  position  of  the  violation  between  stpidxa+1 and
  stpidxa+2.

==========================================================================
= SHORTLY SPEAKING: build a 2D plot of (stp,f) and look at it -  you  will
=                   see where C1 continuity is violated.
==========================================================================

INPUT PARAMETERS:
    state   -   algorithm state

OUTPUT PARAMETERS:
    strrep  -   C1 test #0 "strong" report
    lngrep  -   C1 test #0 "long" report

  -- ALGLIB --
     Copyright 21.11.2018 by Bochkanov Sergey
*************************************************************************/
void minbleicoptguardnonc1test0results(const minbleicstate &state, optguardnonc1test0report &strrep, optguardnonc1test0report &lngrep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Detailed results of the OptGuard integrity check for nonsmoothness test #1

Nonsmoothness (non-C1)  test  #1  studies  individual  components  of  the
gradient computed during line search.

When precise analytic gradient is provided this test is more powerful than
test #0  which  works  with  function  values  and  ignores  user-provided
gradient.  However,  test  #0  becomes  more   powerful   when   numerical
differentiation is employed (in such cases test #1 detects  higher  levels
of numerical noise and becomes too conservative).

This test also tells specific components of the gradient which violate  C1
continuity, which makes it more informative than #0, which just tells that
continuity is violated.

Two reports are returned:
* a "strongest" one, corresponding  to  line   search  which  had  highest
  value of the nonsmoothness indicator
* a "longest" one, corresponding to line search which  had  more  function
  evaluations, and thus is more detailed

In both cases following fields are returned:

* positive - is TRUE  when test flagged suspicious point;  FALSE  if  test
  did not notice anything (in the latter cases fields below are empty).
* vidx - is an index of the variable in [0,N) with nonsmooth derivative
* x0[], d[] - arrays of length N which store initial point  and  direction
  for line search (d[] can be normalized, but does not have to)
* stp[], g[] - arrays of length CNT which store step lengths and  gradient
  values at these points; g[i] is evaluated in  x0+stp[i]*d  and  contains
  vidx-th component of the gradient.
* stpidxa, stpidxb - we  suspect  that  function  violates  C1  continuity
  between steps #stpidxa and #stpidxb (usually we have  stpidxb=stpidxa+3,
  with  most  likely  position  of  the  violation  between  stpidxa+1 and
  stpidxa+2.

==========================================================================
= SHORTLY SPEAKING: build a 2D plot of (stp,f) and look at it -  you  will
=                   see where C1 continuity is violated.
==========================================================================

INPUT PARAMETERS:
    state   -   algorithm state

OUTPUT PARAMETERS:
    strrep  -   C1 test #1 "strong" report
    lngrep  -   C1 test #1 "long" report

  -- ALGLIB --
     Copyright 21.11.2018 by Bochkanov Sergey
*************************************************************************/
void minbleicoptguardnonc1test1results(const minbleicstate &state, optguardnonc1test1report &strrep, optguardnonc1test1report &lngrep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
BLEIC results

INPUT PARAMETERS:
    State   -   algorithm state

OUTPUT PARAMETERS:
    X       -   array[0..N-1], solution
    Rep     -   optimization report. You should check Rep.TerminationType
                in  order  to  distinguish  successful  termination  from
                unsuccessful one:
                * -8    internal integrity control  detected  infinite or
                        NAN   values   in   function/gradient.   Abnormal
                        termination signalled.
                * -3   inconsistent constraints. Feasible point is
                       either nonexistent or too hard to find. Try to
                       restart optimizer with better initial approximation
                *  1   relative function improvement is no more than EpsF.
                *  2   scaled step is no more than EpsX.
                *  4   scaled gradient norm is no more than EpsG.
                *  5   MaxIts steps was taken
                *  8   terminated by user who called minbleicrequesttermination().
                       X contains point which was "current accepted"  when
                       termination request was submitted.
                More information about fields of this  structure  can  be
                found in the comments on MinBLEICReport datatype.

  -- ALGLIB --
     Copyright 28.11.2010 by Bochkanov Sergey
*************************************************************************/
void minbleicresults(const minbleicstate &state, real_1d_array &x, minbleicreport &rep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
BLEIC results

Buffered implementation of MinBLEICResults() which uses pre-allocated buffer
to store X[]. If buffer size is  too  small,  it  resizes  buffer.  It  is
intended to be used in the inner cycles of performance critical algorithms
where array reallocation penalty is too large to be ignored.

  -- ALGLIB --
     Copyright 28.11.2010 by Bochkanov Sergey
*************************************************************************/
void minbleicresultsbuf(const minbleicstate &state, real_1d_array &x, minbleicreport &rep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This subroutine restarts algorithm from new point.
All optimization parameters (including constraints) are left unchanged.

This  function  allows  to  solve multiple  optimization  problems  (which
must have  same number of dimensions) without object reallocation penalty.

INPUT PARAMETERS:
    State   -   structure previously allocated with MinBLEICCreate call.
    X       -   new starting point.

  -- ALGLIB --
     Copyright 28.11.2010 by Bochkanov Sergey
*************************************************************************/
void minbleicrestartfrom(const minbleicstate &state, const real_1d_array &x, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This subroutine submits request for termination of running  optimizer.  It
should be called from user-supplied callback when user decides that it  is
time to "smoothly" terminate optimization process.  As  result,  optimizer
stops at point which was "current accepted" when termination  request  was
submitted and returns error code 8 (successful termination).

INPUT PARAMETERS:
    State   -   optimizer structure

NOTE: after  request  for  termination  optimizer  may   perform   several
      additional calls to user-supplied callbacks. It does  NOT  guarantee
      to stop immediately - it just guarantees that these additional calls
      will be discarded later.

NOTE: calling this function on optimizer which is NOT running will have no
      effect.

NOTE: multiple calls to this function are possible. First call is counted,
      subsequent calls are silently ignored.

  -- ALGLIB --
     Copyright 08.10.2014 by Bochkanov Sergey
*************************************************************************/
void minbleicrequesttermination(const minbleicstate &state, const xparams _xparams = alglib::xdefault);
#endif

#if defined(AE_COMPILE_QPBLEICSOLVER) || !defined(AE_PARTIAL_BUILD)

#endif

#if defined(AE_COMPILE_MINQP) || !defined(AE_PARTIAL_BUILD)
/*************************************************************************
                    CONSTRAINED QUADRATIC PROGRAMMING

The subroutine creates QP optimizer. After initial creation,  it  contains
default optimization problem with zero quadratic and linear terms  and  no
constraints. You should set quadratic/linear terms with calls to functions
provided by MinQP subpackage.

You should also choose appropriate QP solver and set it  and  its stopping
criteria by means of MinQPSetAlgo??????() function. Then, you should start
solution process by means of MinQPOptimize() call. Solution itself can  be
obtained with MinQPResults() function.

Following solvers are recommended:
* QuickQP for dense problems with box-only constraints (or no constraints
  at all)
* QP-BLEIC for dense/sparse problems with moderate (up to 50) number of
  general linear constraints
* DENSE-AUL-QP for dense problems with any (small or large) number of
  general linear constraints

INPUT PARAMETERS:
    N       -   problem size

OUTPUT PARAMETERS:
    State   -   optimizer with zero quadratic/linear terms
                and no constraints

  -- ALGLIB --
     Copyright 11.01.2011 by Bochkanov Sergey
*************************************************************************/
void minqpcreate(const ae_int_t n, minqpstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets linear term for QP solver.

By default, linear term is zero.

INPUT PARAMETERS:
    State   -   structure which stores algorithm state
    B       -   linear term, array[N].

  -- ALGLIB --
     Copyright 11.01.2011 by Bochkanov Sergey
*************************************************************************/
void minqpsetlinearterm(const minqpstate &state, const real_1d_array &b, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This  function  sets  dense  quadratic  term  for  QP solver. By  default,
quadratic term is zero.

SUPPORT BY QP SOLVERS:

Dense quadratic term can be handled by following QP solvers:
* QuickQP
* BLEIC-QP
* Dense-AUL-QP

IMPORTANT:

This solver minimizes following  function:
    f(x) = 0.5*x'*A*x + b'*x.
Note that quadratic term has 0.5 before it. So if  you  want  to  minimize
    f(x) = x^2 + x
you should rewrite your problem as follows:
    f(x) = 0.5*(2*x^2) + x
and your matrix A will be equal to [[2.0]], not to [[1.0]]

INPUT PARAMETERS:
    State   -   structure which stores algorithm state
    A       -   matrix, array[N,N]
    IsUpper -   (optional) storage type:
                * if True, symmetric matrix  A  is  given  by  its  upper
                  triangle, and the lower triangle isn't used
                * if False, symmetric matrix  A  is  given  by  its lower
                  triangle, and the upper triangle isn't used
                * if not given, both lower and upper  triangles  must  be
                  filled.

  -- ALGLIB --
     Copyright 11.01.2011 by Bochkanov Sergey
*************************************************************************/
void minqpsetquadraticterm(const minqpstate &state, const real_2d_array &a, const bool isupper, const xparams _xparams = alglib::xdefault);
void minqpsetquadraticterm(const minqpstate &state, const real_2d_array &a, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This  function  sets  sparse  quadratic  term  for  QP solver. By default,
quadratic  term  is  zero.  This  function  overrides  previous  calls  to
minqpsetquadraticterm() or minqpsetquadratictermsparse().

SUPPORT BY QP SOLVERS:

Sparse quadratic term can be handled by following QP solvers:
* QuickQP
* BLEIC-QP
* Dense-AUL-QP (internally converts sparse matrix to dense format)

IMPORTANT:

This solver minimizes following  function:
    f(x) = 0.5*x'*A*x + b'*x.
Note that quadratic term has 0.5 before it. So if  you  want  to  minimize
    f(x) = x^2 + x
you should rewrite your problem as follows:
    f(x) = 0.5*(2*x^2) + x
and your matrix A will be equal to [[2.0]], not to [[1.0]]

INPUT PARAMETERS:
    State   -   structure which stores algorithm state
    A       -   matrix, array[N,N]
    IsUpper -   (optional) storage type:
                * if True, symmetric matrix  A  is  given  by  its  upper
                  triangle, and the lower triangle isn't used
                * if False, symmetric matrix  A  is  given  by  its lower
                  triangle, and the upper triangle isn't used
                * if not given, both lower and upper  triangles  must  be
                  filled.

  -- ALGLIB --
     Copyright 11.01.2011 by Bochkanov Sergey
*************************************************************************/
void minqpsetquadratictermsparse(const minqpstate &state, const sparsematrix &a, const bool isupper, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets starting point for QP solver. It is useful to have
good initial approximation to the solution, because it will increase
speed of convergence and identification of active constraints.

INPUT PARAMETERS:
    State   -   structure which stores algorithm state
    X       -   starting point, array[N].

  -- ALGLIB --
     Copyright 11.01.2011 by Bochkanov Sergey
*************************************************************************/
void minqpsetstartingpoint(const minqpstate &state, const real_1d_array &x, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This  function sets origin for QP solver. By default, following QP program
is solved:

    min(0.5*x'*A*x+b'*x)

This function allows to solve different problem:

    min(0.5*(x-x_origin)'*A*(x-x_origin)+b'*(x-x_origin))

Specification of non-zero origin affects function being minimized, but not
constraints. Box and  linear  constraints  are  still  calculated  without
origin.

INPUT PARAMETERS:
    State   -   structure which stores algorithm state
    XOrigin -   origin, array[N].

  -- ALGLIB --
     Copyright 11.01.2011 by Bochkanov Sergey
*************************************************************************/
void minqpsetorigin(const minqpstate &state, const real_1d_array &xorigin, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets scaling coefficients.

ALGLIB optimizers use scaling matrices to test stopping  conditions  (step
size and gradient are scaled before comparison  with  tolerances)  and  as
preconditioner.

Scale of the I-th variable is a translation invariant measure of:
a) "how large" the variable is
b) how large the step should be to make significant changes in the
   function

If you do not know how to choose scales of your variables, you can:
* read www.alglib.net/optimization/scaling.php article
* use minqpsetscaleautodiag(), which calculates scale  using  diagonal  of
  the  quadratic  term:  S  is  set to 1/sqrt(diag(A)), which works well
  sometimes.

INPUT PARAMETERS:
    State   -   structure stores algorithm state
    S       -   array[N], non-zero scaling coefficients
                S[i] may be negative, sign doesn't matter.

  -- ALGLIB --
     Copyright 14.01.2011 by Bochkanov Sergey
*************************************************************************/
void minqpsetscale(const minqpstate &state, const real_1d_array &s, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets automatic evaluation of variable scaling.

IMPORTANT: this function works only for  matrices  with positive  diagonal
           elements! Zero or negative elements will  result  in  -9  error
           code  being  returned.  Specify  scale  vector  manually   with
           minqpsetscale() in such cases.

ALGLIB optimizers use scaling matrices to test stopping  conditions  (step
size and gradient are scaled before comparison  with  tolerances)  and  as
preconditioner.

The  best  way  to  set  scaling  is  to manually specify variable scales.
However, sometimes you just need quick-and-dirty solution  -  either  when
you perform fast prototyping, or when you know your problem well  and  you
are 100% sure that this quick solution is robust enough in your case.

One such solution is to evaluate scale of I-th variable as 1/Sqrt(A[i,i]),
where A[i,i] is an I-th diagonal element of the quadratic term.

Such approach works well sometimes, but you have to be careful here.

INPUT PARAMETERS:
    State   -   structure stores algorithm state

  -- ALGLIB --
     Copyright 26.12.2017 by Bochkanov Sergey
*************************************************************************/
void minqpsetscaleautodiag(const minqpstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function tells solver to use BLEIC-based algorithm and sets  stopping
criteria for the algorithm.

This algorithm is fast  enough  for large-scale  problems  with  following
properties:
a) feasible initial point, moderate amount of general linear constraints
b) arbitrary (can be infeasible) initial point, small  amount  of  general
   linear constraints (say, hundred or less)

If you solve large-scale QP problem with many inequality  constraints  and
without initial feasibility guarantees, consider  using  DENSE-AUL  solver
instead. Initial feasibility detection stage by BLEIC may take too long on
such problems.

ALGORITHM FEATURES:

* supports dense and sparse QP problems
* supports box and general linear equality/inequality constraints
* can solve all types of problems  (convex,  semidefinite,  nonconvex)  as
  long as they are bounded from below under constraints.
  Say, it is possible to solve "min{-x^2} subject to -1<=x<=+1".
  Of course, global  minimum  is found only  for  positive  definite   and
  semidefinite  problems.  As  for indefinite ones - only local minimum is
  found.

ALGORITHM OUTLINE:

* BLEIC-QP solver is just a driver function for MinBLEIC solver; it solves
  quadratic  programming   problem   as   general   linearly   constrained
  optimization problem, which is solved by means of BLEIC solver  (part of
  ALGLIB, active set method).

ALGORITHM LIMITATIONS:
* This algorithm is inefficient on  problems with hundreds  and  thousands
  of general inequality constraints and infeasible initial point.  Initial
  feasibility detection stage may take too long on such constraint sets.
  Consider using DENSE-AUL instead.
* unlike QuickQP solver, this algorithm does not perform Newton steps  and
  does not use Level 3 BLAS. Being general-purpose active set  method,  it
  can activate constraints only one-by-one. Thus, its performance is lower
  than that of QuickQP.
* its precision is also a bit  inferior  to  that  of   QuickQP.  BLEIC-QP
  performs only LBFGS steps (no Newton steps), which are good at detecting
  neighborhood of the solution, buy needs many iterations to find solution
  with more than 6 digits of precision.

INPUT PARAMETERS:
    State   -   structure which stores algorithm state
    EpsG    -   >=0
                The  subroutine  finishes  its  work   if   the  condition
                |v|<EpsG is satisfied, where:
                * |.| means Euclidian norm
                * v - scaled constrained gradient vector, v[i]=g[i]*s[i]
                * g - gradient
                * s - scaling coefficients set by MinQPSetScale()
    EpsF    -   >=0
                The  subroutine  finishes its work if exploratory steepest
                descent  step  on  k+1-th iteration  satisfies   following
                condition:  |F(k+1)-F(k)|<=EpsF*max{|F(k)|,|F(k+1)|,1}
    EpsX    -   >=0
                The  subroutine  finishes its work if exploratory steepest
                descent  step  on  k+1-th iteration  satisfies   following
                condition:
                * |.| means Euclidian norm
                * v - scaled step vector, v[i]=dx[i]/s[i]
                * dx - step vector, dx=X(k+1)-X(k)
                * s - scaling coefficients set by MinQPSetScale()
    MaxIts  -   maximum number of iterations. If MaxIts=0, the  number  of
                iterations is unlimited. NOTE: this  algorithm uses  LBFGS
                iterations,  which  are  relatively  cheap,  but   improve
                function value only a bit. So you will need many iterations
                to converge - from 0.1*N to 10*N, depending  on  problem's
                condition number.

IT IS VERY IMPORTANT TO CALL MinQPSetScale() WHEN YOU USE THIS  ALGORITHM
BECAUSE ITS STOPPING CRITERIA ARE SCALE-DEPENDENT!

Passing EpsG=0, EpsF=0 and EpsX=0 and MaxIts=0 (simultaneously) will lead
to automatic stopping criterion selection (presently it is  small    step
length, but it may change in the future versions of ALGLIB).

  -- ALGLIB --
     Copyright 11.01.2011 by Bochkanov Sergey
*************************************************************************/
void minqpsetalgobleic(const minqpstate &state, const double epsg, const double epsf, const double epsx, const ae_int_t maxits, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function tells QP solver to use Dense-AUL algorithm and sets stopping
criteria for the algorithm.

ALGORITHM FEATURES:

* supports  box  and  dense/sparse  general   linear   equality/inequality
  constraints
* convergence is theoretically proved for positive-definite  (convex)   QP
  problems. Semidefinite and non-convex problems can be solved as long  as
  they  are   bounded  from  below  under  constraints,  although  without
  theoretical guarantees.
* this solver is better than QP-BLEIC on problems  with  large  number  of
  general linear constraints. It better handles infeasible initial points.

ALGORITHM OUTLINE:

* this  algorithm   is   an   augmented   Lagrangian   method  with  dense
  preconditioner (hence  its  name).  It  is  similar  to  barrier/penalty
  methods, but much more precise and faster.
* it performs several outer iterations in order to refine  values  of  the
  Lagrange multipliers. Single outer  iteration  is  a  solution  of  some
  unconstrained optimization problem: first  it  performs  dense  Cholesky
  factorization of the Hessian in order to build preconditioner  (adaptive
  regularization is applied to enforce positive  definiteness),  and  then
  it uses L-BFGS optimizer to solve optimization problem.
* typically you need about 5-10 outer iterations to converge to solution

ALGORITHM LIMITATIONS:

* because dense Cholesky driver is used, this algorithm has O(N^2)  memory
  requirements and O(OuterIterations*N^3) minimum running time.  From  the
  practical  point  of  view,  it  limits  its  applicability  by  several
  thousands of variables.
  From  the  other  side,  variables  count  is  the most limiting factor,
  and dependence on constraint count is  much  more  lower. Assuming  that
  constraint matrix is sparse, it may handle tens of thousands  of general
  linear constraints.

INPUT PARAMETERS:
    State   -   structure which stores algorithm state
    EpsX    -   >=0, stopping criteria for inner optimizer.
                Inner  iterations  are  stopped  when  step  length  (with
                variable scaling being applied) is less than EpsX.
                See  minqpsetscale()  for  more  information  on  variable
                scaling.
    Rho     -   penalty coefficient, Rho>0:
                * large enough  that  algorithm  converges  with   desired
                  precision.
                * not TOO large to prevent ill-conditioning
                * recommended values are 100, 1000 or 10000
    ItsCnt  -   number of outer iterations:
                * recommended values: 10-15 (although  in  most  cases  it
                  converges within 5 iterations, you may need a  few  more
                  to be sure).
                * ItsCnt=0 means that small number of outer iterations  is
                  automatically chosen (10 iterations in current version).
                * ItsCnt=1 means that AUL algorithm performs just as usual
                  penalty method.
                * ItsCnt>1 means that  AUL  algorithm  performs  specified
                  number of outer iterations

IT IS VERY IMPORTANT TO CALL minqpsetscale() WHEN YOU USE THIS  ALGORITHM
BECAUSE ITS CONVERGENCE PROPERTIES AND STOPPING CRITERIA ARE SCALE-DEPENDENT!

NOTE: Passing  EpsX=0  will  lead  to  automatic  step  length  selection
      (specific step length chosen may change in the future  versions  of
      ALGLIB, so it is better to specify step length explicitly).

  -- ALGLIB --
     Copyright 20.08.2016 by Bochkanov Sergey
*************************************************************************/
void minqpsetalgodenseaul(const minqpstate &state, const double epsx, const double rho, const ae_int_t itscnt, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function tells solver to use QuickQP  algorithm:  special  extra-fast
algorithm for problems with box-only constrants. It may  solve  non-convex
problems as long as they are bounded from below under constraints.

ALGORITHM FEATURES:
* many times (from 5x to 50x!) faster than BLEIC-based QP solver; utilizes
  accelerated methods for activation of constraints.
* supports dense and sparse QP problems
* supports ONLY box constraints; general linear constraints are NOT
  supported by this solver
* can solve all types of problems  (convex,  semidefinite,  nonconvex)  as
  long as they are bounded from below under constraints.
  Say, it is possible to solve "min{-x^2} subject to -1<=x<=+1".
  In convex/semidefinite case global minimum  is  returned,  in  nonconvex
  case - algorithm returns one of the local minimums.

ALGORITHM OUTLINE:

* algorithm  performs  two kinds of iterations: constrained CG  iterations
  and constrained Newton iterations
* initially it performs small number of constrained CG  iterations,  which
  can efficiently activate/deactivate multiple constraints
* after CG phase algorithm tries to calculate Cholesky  decomposition  and
  to perform several constrained Newton steps. If  Cholesky  decomposition
  failed (matrix is indefinite even under constraints),  we  perform  more
  CG iterations until we converge to such set of constraints  that  system
  matrix becomes  positive  definite.  Constrained  Newton  steps  greatly
  increase convergence speed and precision.
* algorithm interleaves CG and Newton iterations which  allows  to  handle
  indefinite matrices (CG phase) and quickly converge after final  set  of
  constraints is found (Newton phase). Combination of CG and Newton phases
  is called "outer iteration".
* it is possible to turn off Newton  phase  (beneficial  for  semidefinite
  problems - Cholesky decomposition will fail too often)

ALGORITHM LIMITATIONS:

* algorithm does not support general  linear  constraints;  only  box ones
  are supported
* Cholesky decomposition for sparse problems  is  performed  with  Skyline
  Cholesky solver, which is intended for low-profile matrices. No profile-
  reducing reordering of variables is performed in this version of ALGLIB.
* problems with near-zero negative eigenvalues (or exacty zero  ones)  may
  experience about 2-3x performance penalty. The reason is  that  Cholesky
  decomposition can not be performed until we identify directions of  zero
  and negative curvature and activate corresponding boundary constraints -
  but we need a lot of trial and errors because these directions  are hard
  to notice in the matrix spectrum.
  In this case you may turn off Newton phase of algorithm.
  Large negative eigenvalues  are  not  an  issue,  so  highly  non-convex
  problems can be solved very efficiently.

INPUT PARAMETERS:
    State   -   structure which stores algorithm state
    EpsG    -   >=0
                The  subroutine  finishes  its  work   if   the  condition
                |v|<EpsG is satisfied, where:
                * |.| means Euclidian norm
                * v - scaled constrained gradient vector, v[i]=g[i]*s[i]
                * g - gradient
                * s - scaling coefficients set by MinQPSetScale()
    EpsF    -   >=0
                The  subroutine  finishes its work if exploratory steepest
                descent  step  on  k+1-th iteration  satisfies   following
                condition:  |F(k+1)-F(k)|<=EpsF*max{|F(k)|,|F(k+1)|,1}
    EpsX    -   >=0
                The  subroutine  finishes its work if exploratory steepest
                descent  step  on  k+1-th iteration  satisfies   following
                condition:
                * |.| means Euclidian norm
                * v - scaled step vector, v[i]=dx[i]/s[i]
                * dx - step vector, dx=X(k+1)-X(k)
                * s - scaling coefficients set by MinQPSetScale()
    MaxOuterIts-maximum number of OUTER iterations.  One  outer  iteration
                includes some amount of CG iterations (from 5 to  ~N)  and
                one or several (usually small amount) Newton steps.  Thus,
                one outer iteration has high cost, but can greatly  reduce
                funcation value.
                Use 0 if you do not want to limit number of outer iterations.
    UseNewton-  use Newton phase or not:
                * Newton phase improves performance of  positive  definite
                  dense problems (about 2 times improvement can be observed)
                * can result in some performance penalty  on  semidefinite
                  or slightly negative definite  problems  -  each  Newton
                  phase will bring no improvement (Cholesky failure),  but
                  still will require computational time.
                * if you doubt, you can turn off this  phase  -  optimizer
                  will retain its most of its high speed.

IT IS VERY IMPORTANT TO CALL MinQPSetScale() WHEN YOU USE THIS  ALGORITHM
BECAUSE ITS STOPPING CRITERIA ARE SCALE-DEPENDENT!

Passing EpsG=0, EpsF=0 and EpsX=0 and MaxIts=0 (simultaneously) will lead
to automatic stopping criterion selection (presently it is  small    step
length, but it may change in the future versions of ALGLIB).

  -- ALGLIB --
     Copyright 22.05.2014 by Bochkanov Sergey
*************************************************************************/
void minqpsetalgoquickqp(const minqpstate &state, const double epsg, const double epsf, const double epsx, const ae_int_t maxouterits, const bool usenewton, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets box constraints for QP solver

Box constraints are inactive by default (after  initial  creation).  After
being  set,  they  are  preserved until explicitly turned off with another
SetBC() call.

All QP solvers may handle box constraints.

INPUT PARAMETERS:
    State   -   structure stores algorithm state
    BndL    -   lower bounds, array[N].
                If some (all) variables are unbounded, you may specify
                very small number or -INF (latter is recommended because
                it will allow solver to use better algorithm).
    BndU    -   upper bounds, array[N].
                If some (all) variables are unbounded, you may specify
                very large number or +INF (latter is recommended because
                it will allow solver to use better algorithm).

NOTE: it is possible to specify BndL[i]=BndU[i]. In this case I-th
variable will be "frozen" at X[i]=BndL[i]=BndU[i].

  -- ALGLIB --
     Copyright 11.01.2011 by Bochkanov Sergey
*************************************************************************/
void minqpsetbc(const minqpstate &state, const real_1d_array &bndl, const real_1d_array &bndu, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets dense linear constraints for QP optimizer.

This  function  overrides  results  of  previous  calls  to  minqpsetlc(),
minqpsetlcsparse() and minqpsetlcmixed().  After  call  to  this  function
sparse constraints are dropped, and you have only those constraints  which
were specified in the present call.

If you want  to  specify  mixed  (with  dense  and  sparse  terms)  linear
constraints, you should call minqpsetlcmixed().

SUPPORT BY QP SOLVERS:

Following QP solvers can handle dense linear constraints:
* BLEIC-QP          -   handles them  with  high  precision,  but  may  be
                        inefficient for problems with hundreds of constraints
* Dense-AUL-QP      -   handles them with moderate precision (approx. 10^-6),
                        may efficiently handle thousands of constraints.

Following QP solvers can NOT handle dense linear constraints:
* QuickQP           -   can not handle general linear constraints

INPUT PARAMETERS:
    State   -   structure previously allocated with MinQPCreate call.
    C       -   linear constraints, array[K,N+1].
                Each row of C represents one constraint, either equality
                or inequality (see below):
                * first N elements correspond to coefficients,
                * last element corresponds to the right part.
                All elements of C (including right part) must be finite.
    CT      -   type of constraints, array[K]:
                * if CT[i]>0, then I-th constraint is C[i,*]*x >= C[i,n+1]
                * if CT[i]=0, then I-th constraint is C[i,*]*x  = C[i,n+1]
                * if CT[i]<0, then I-th constraint is C[i,*]*x <= C[i,n+1]
    K       -   number of equality/inequality constraints, K>=0:
                * if given, only leading K elements of C/CT are used
                * if not given, automatically determined from sizes of C/CT

NOTE 1: linear (non-bound) constraints are satisfied only approximately  -
        there always exists some violation due  to  numerical  errors  and
        algorithmic limitations (BLEIC-QP solver is most  precise,  AUL-QP
        solver is less precise).

  -- ALGLIB --
     Copyright 19.06.2012 by Bochkanov Sergey
*************************************************************************/
void minqpsetlc(const minqpstate &state, const real_2d_array &c, const integer_1d_array &ct, const ae_int_t k, const xparams _xparams = alglib::xdefault);
void minqpsetlc(const minqpstate &state, const real_2d_array &c, const integer_1d_array &ct, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets sparse linear constraints for QP optimizer.

This  function  overrides  results  of  previous  calls  to  minqpsetlc(),
minqpsetlcsparse() and minqpsetlcmixed().  After  call  to  this  function
dense constraints are dropped, and you have only those  constraints  which
were specified in the present call.

If you want  to  specify  mixed  (with  dense  and  sparse  terms)  linear
constraints, you should call minqpsetlcmixed().

SUPPORT BY QP SOLVERS:

Following QP solvers can handle sparse linear constraints:
* BLEIC-QP          -   handles them  with  high  precision,  but can  not
                        utilize their sparsity - sparse constraint  matrix
                        is silently converted to dense  format.  Thus,  it
                        may be inefficient for problems with  hundreds  of
                        constraints.
* Dense-AUL-QP      -   although this solver uses dense linear algebra  to
                        calculate   Cholesky   preconditioner,   it    may
                        efficiently  handle  sparse  constraints.  It  may
                        solve problems  with  hundreds  and  thousands  of
                        constraints. The only drawback is  that  precision
                        of constraint handling is typically within 1E-4...
                        ..1E-6 range.

Following QP solvers can NOT handle sparse linear constraints:
* QuickQP           -   can not handle general linear constraints

INPUT PARAMETERS:
    State   -   structure previously allocated with MinQPCreate call.
    C       -   linear  constraints,  sparse  matrix  with  dimensions  at
                least [K,N+1]. If matrix has  larger  size,  only  leading
                Kx(N+1) rectangle is used.
                Each row of C represents one constraint, either equality
                or inequality (see below):
                * first N elements correspond to coefficients,
                * last element corresponds to the right part.
                All elements of C (including right part) must be finite.
    CT      -   type of constraints, array[K]:
                * if CT[i]>0, then I-th constraint is C[i,*]*x >= C[i,n+1]
                * if CT[i]=0, then I-th constraint is C[i,*]*x  = C[i,n+1]
                * if CT[i]<0, then I-th constraint is C[i,*]*x <= C[i,n+1]
    K       -   number of equality/inequality constraints, K>=0

NOTE 1: linear (non-bound) constraints are satisfied only approximately  -
        there always exists some violation due  to  numerical  errors  and
        algorithmic limitations (BLEIC-QP solver is most  precise,  AUL-QP
        solver is less precise).

  -- ALGLIB --
     Copyright 22.08.2016 by Bochkanov Sergey
*************************************************************************/
void minqpsetlcsparse(const minqpstate &state, const sparsematrix &c, const integer_1d_array &ct, const ae_int_t k, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets mixed linear constraints, which include a set of  dense
rows, and a set of sparse rows.

This  function  overrides  results  of  previous  calls  to  minqpsetlc(),
minqpsetlcsparse() and minqpsetlcmixed().

This function may be useful if constraint matrix includes large number  of
both types of rows - dense and sparse. If you have just a few sparse rows,
you  may  represent  them  in  dense  format  without loosing performance.
Similarly, if you have just a few dense rows, you may store them in sparse
format with almost same performance.

SUPPORT BY QP SOLVERS:

Following QP solvers can handle mixed dense/sparse linear constraints:
* BLEIC-QP          -   handles them  with  high  precision,  but can  not
                        utilize their sparsity - sparse constraint  matrix
                        is silently converted to dense  format.  Thus,  it
                        may be inefficient for problems with  hundreds  of
                        constraints.
* Dense-AUL-QP      -   although this solver uses dense linear algebra  to
                        calculate   Cholesky   preconditioner,   it    may
                        efficiently  handle  sparse  constraints.  It  may
                        solve problems  with  hundreds  and  thousands  of
                        constraints. The only drawback is  that  precision
                        of constraint handling is typically within 1E-4...
                        ..1E-6 range.

Following QP solvers can NOT handle mixed linear constraints:
* QuickQP           -   can not handle general linear constraints at all

INPUT PARAMETERS:
    State   -   structure previously allocated with MinQPCreate call.
    DenseC  -   dense linear constraints, array[K,N+1].
                Each row of DenseC represents one constraint, either equality
                or inequality (see below):
                * first N elements correspond to coefficients,
                * last element corresponds to the right part.
                All elements of DenseC (including right part) must be finite.
    DenseCT -   type of constraints, array[K]:
                * if DenseCT[i]>0, then I-th constraint is DenseC[i,*]*x >= DenseC[i,n+1]
                * if DenseCT[i]=0, then I-th constraint is DenseC[i,*]*x  = DenseC[i,n+1]
                * if DenseCT[i]<0, then I-th constraint is DenseC[i,*]*x <= DenseC[i,n+1]
    DenseK  -   number of equality/inequality constraints, DenseK>=0
    SparseC -   linear  constraints,  sparse  matrix  with  dimensions  at
                least [SparseK,N+1]. If matrix has  larger  size,  only  leading
                SPARSEKx(N+1) rectangle is used.
                Each row of C represents one constraint, either equality
                or inequality (see below):
                * first N elements correspond to coefficients,
                * last element corresponds to the right part.
                All elements of C (including right part) must be finite.
    SparseCT-   type of sparse constraints, array[K]:
                * if SparseCT[i]>0, then I-th constraint is SparseC[i,*]*x >= SparseC[i,n+1]
                * if SparseCT[i]=0, then I-th constraint is SparseC[i,*]*x  = SparseC[i,n+1]
                * if SparseCT[i]<0, then I-th constraint is SparseC[i,*]*x <= SparseC[i,n+1]
    SparseK -   number of sparse equality/inequality constraints, K>=0

NOTE 1: linear (non-bound) constraints are satisfied only approximately  -
        there always exists some violation due  to  numerical  errors  and
        algorithmic limitations (BLEIC-QP solver is most  precise,  AUL-QP
        solver is less precise).

  -- ALGLIB --
     Copyright 22.08.2016 by Bochkanov Sergey
*************************************************************************/
void minqpsetlcmixed(const minqpstate &state, const real_2d_array &densec, const integer_1d_array &densect, const ae_int_t densek, const sparsematrix &sparsec, const integer_1d_array &sparsect, const ae_int_t sparsek, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function solves quadratic programming problem.

Prior to calling this function you should choose solver by means of one of
the following functions:

* minqpsetalgoquickqp()     - for QuickQP solver
* minqpsetalgobleic()       - for BLEIC-QP solver
* minqpsetalgodenseaul()    - for Dense-AUL-QP solver

These functions also allow you to control stopping criteria of the solver.
If you did not set solver,  MinQP  subpackage  will  automatically  select
solver for your problem and will run it with default stopping criteria.

However, it is better to set explicitly solver and its stopping criteria.

INPUT PARAMETERS:
    State   -   algorithm state

You should use MinQPResults() function to access results after calls
to this function.

  -- ALGLIB --
     Copyright 11.01.2011 by Bochkanov Sergey.
     Special thanks to Elvira Illarionova  for  important  suggestions  on
     the linearly constrained QP algorithm.
*************************************************************************/
void minqpoptimize(const minqpstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
QP solver results

INPUT PARAMETERS:
    State   -   algorithm state

OUTPUT PARAMETERS:
    X       -   array[0..N-1], solution.
                This array is allocated and initialized only when
                Rep.TerminationType parameter is positive (success).
    Rep     -   optimization report. You should check Rep.TerminationType,
                which contains completion code, and you may check  another
                fields which contain another information  about  algorithm
                functioning.

                Failure codes returned by algorithm are:
                * -9    failure of the automatic scale evaluation:  one of
                        the diagonal elements of  the  quadratic  term  is
                        non-positive.  Specify variable scales manually!
                * -5    inappropriate solver was used:
                        * QuickQP solver for problem with  general  linear
                          constraints
                * -4    BLEIC-QP/QuickQP   solver    found   unconstrained
                        direction  of   negative  curvature  (function  is
                        unbounded from below even under constraints),   no
                        meaningful minimum can be found.
                * -3    inconsistent constraints (or maybe  feasible point
                        is too  hard  to  find).  If  you  are  sure  that
                        constraints are feasible, try to restart optimizer
                        with better initial approximation.

                Completion codes specific for Cholesky algorithm:
                *  4   successful completion

                Completion codes specific for BLEIC/QuickQP algorithms:
                *  1   relative function improvement is no more than EpsF.
                *  2   scaled step is no more than EpsX.
                *  4   scaled gradient norm is no more than EpsG.
                *  5   MaxIts steps was taken

  -- ALGLIB --
     Copyright 11.01.2011 by Bochkanov Sergey
*************************************************************************/
void minqpresults(const minqpstate &state, real_1d_array &x, minqpreport &rep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
QP results

Buffered implementation of MinQPResults() which uses pre-allocated  buffer
to store X[]. If buffer size is  too  small,  it  resizes  buffer.  It  is
intended to be used in the inner cycles of performance critical algorithms
where array reallocation penalty is too large to be ignored.

  -- ALGLIB --
     Copyright 11.01.2011 by Bochkanov Sergey
*************************************************************************/
void minqpresultsbuf(const minqpstate &state, real_1d_array &x, minqpreport &rep, const xparams _xparams = alglib::xdefault);
#endif

#if defined(AE_COMPILE_REVISEDDUALSIMPLEX) || !defined(AE_PARTIAL_BUILD)

#endif

#if defined(AE_COMPILE_MINLP) || !defined(AE_PARTIAL_BUILD)
/*************************************************************************
                            LINEAR PROGRAMMING

The subroutine creates LP  solver.  After  initial  creation  it  contains
default optimization problem with zero cost vector and all variables being
fixed to zero values and no constraints.

In order to actually solve something you should:
* set cost vector with minlpsetcost()
* set variable bounds with minlpsetbc() or minlpsetbcall()
* specify constraint matrix with one of the following functions:
  [*] minlpsetlc()        for dense one-sided constraints
  [*] minlpsetlc2dense()  for dense two-sided constraints
  [*] minlpsetlc2()       for sparse two-sided constraints
  [*] minlpaddlc2dense()  to add one dense row to constraint matrix
  [*] minlpaddlc2()       to add one row to constraint matrix (compressed format)
* call minlpoptimize() to run the solver and  minlpresults()  to  get  the
  solution vector and additional information.

Presently  this  optimizer  supports  only  revised  simplex   method   as
underlying solver. DSE pricing and bounds flipping ratio  test  (aka  long
dual step) are supported. Large-scale sparse LU solver with  Forest-Tomlin
is used internally as linear algebra driver.

Future releases of ALGLIB may introduce other solvers.

INPUT PARAMETERS:
    N       -   problem size

OUTPUT PARAMETERS:
    State   -   optimizer in the default state

  -- ALGLIB --
     Copyright 19.07.2018 by Bochkanov Sergey
*************************************************************************/
void minlpcreate(const ae_int_t n, minlpstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets cost term for LP solver.

By default, cost term is zero.

INPUT PARAMETERS:
    State   -   structure which stores algorithm state
    C       -   cost term, array[N].

  -- ALGLIB --
     Copyright 19.07.2018 by Bochkanov Sergey
*************************************************************************/
void minlpsetcost(const minlpstate &state, const real_1d_array &c, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets scaling coefficients.

ALGLIB optimizers use scaling matrices to test stopping  conditions and as
preconditioner.

Scale of the I-th variable is a translation invariant measure of:
a) "how large" the variable is
b) how large the step should be to make significant changes in the
   function

INPUT PARAMETERS:
    State   -   structure stores algorithm state
    S       -   array[N], non-zero scaling coefficients
                S[i] may be negative, sign doesn't matter.

  -- ALGLIB --
     Copyright 19.07.2018 by Bochkanov Sergey
*************************************************************************/
void minlpsetscale(const minlpstate &state, const real_1d_array &s, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets box constraints for LP solver (all variables  at  once,
different constraints for different variables).

The default state of constraints is to have all variables fixed  at  zero.
You have to overwrite it by your own constraint vector. Constraint  status
is preserved until constraints are  explicitly  overwritten  with  another
minlpsetbc()  call,   overwritten   with  minlpsetbcall(),  or   partially
overwritten with minlmsetbci() call.

Following types of constraints are supported:

    DESCRIPTION         CONSTRAINT              HOW TO SPECIFY
    fixed variable      x[i]=Bnd[i]             BndL[i]=BndU[i]
    lower bound         BndL[i]<=x[i]           BndU[i]=+INF
    upper bound         x[i]<=BndU[i]           BndL[i]=-INF
    range               BndL[i]<=x[i]<=BndU[i]  ...
    free variable       -                       BndL[I]=-INF, BndU[I]+INF

INPUT PARAMETERS:
    State   -   structure stores algorithm state
    BndL    -   lower bounds, array[N].
    BndU    -   upper bounds, array[N].

NOTE: infinite values can be specified by means of Double.PositiveInfinity
      and  Double.NegativeInfinity  (in  C#)  and  alglib::fp_posinf   and
      alglib::fp_neginf (in C++).

NOTE: you may replace infinities by very small/very large values,  but  it
      is not recommended because large numbers may introduce large numerical
      errors in the algorithm.

NOTE: if constraints for all variables are same you may use minlpsetbcall()
      which allows to specify constraints without using arrays.

NOTE: BndL>BndU will result in LP problem being recognized as infeasible.

  -- ALGLIB --
     Copyright 19.07.2018 by Bochkanov Sergey
*************************************************************************/
void minlpsetbc(const minlpstate &state, const real_1d_array &bndl, const real_1d_array &bndu, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets box constraints for LP solver (all variables  at  once,
same constraints for all variables)

The default state of constraints is to have all variables fixed  at  zero.
You have to overwrite it by your own constraint vector. Constraint  status
is preserved until constraints are  explicitly  overwritten  with  another
minlpsetbc() call or partially overwritten with minlpsetbcall().

Following types of constraints are supported:

    DESCRIPTION         CONSTRAINT              HOW TO SPECIFY
    fixed variable      x[i]=Bnd[i]             BndL[i]=BndU[i]
    lower bound         BndL[i]<=x[i]           BndU[i]=+INF
    upper bound         x[i]<=BndU[i]           BndL[i]=-INF
    range               BndL[i]<=x[i]<=BndU[i]  ...
    free variable       -                       BndL[I]=-INF, BndU[I]+INF

INPUT PARAMETERS:
    State   -   structure stores algorithm state
    BndL    -   lower bound, same for all variables
    BndU    -   upper bound, same for all variables

NOTE: infinite values can be specified by means of Double.PositiveInfinity
      and  Double.NegativeInfinity  (in  C#)  and  alglib::fp_posinf   and
      alglib::fp_neginf (in C++).

NOTE: you may replace infinities by very small/very large values,  but  it
      is not recommended because large numbers may introduce large numerical
      errors in the algorithm.

NOTE: minlpsetbc() can  be  used  to  specify  different  constraints  for
      different variables.

NOTE: BndL>BndU will result in LP problem being recognized as infeasible.

  -- ALGLIB --
     Copyright 19.07.2018 by Bochkanov Sergey
*************************************************************************/
void minlpsetbcall(const minlpstate &state, const double bndl, const double bndu, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets box constraints for I-th variable (other variables are
not modified).

The default state of constraints is to have all variables fixed  at  zero.
You have to overwrite it by your own constraint vector.

Following types of constraints are supported:

    DESCRIPTION         CONSTRAINT              HOW TO SPECIFY
    fixed variable      x[i]=Bnd[i]             BndL[i]=BndU[i]
    lower bound         BndL[i]<=x[i]           BndU[i]=+INF
    upper bound         x[i]<=BndU[i]           BndL[i]=-INF
    range               BndL[i]<=x[i]<=BndU[i]  ...
    free variable       -                       BndL[I]=-INF, BndU[I]+INF

INPUT PARAMETERS:
    State   -   structure stores algorithm state
    I       -   variable index, in [0,N)
    BndL    -   lower bound for I-th variable
    BndU    -   upper bound for I-th variable

NOTE: infinite values can be specified by means of Double.PositiveInfinity
      and  Double.NegativeInfinity  (in  C#)  and  alglib::fp_posinf   and
      alglib::fp_neginf (in C++).

NOTE: you may replace infinities by very small/very large values,  but  it
      is not recommended because large numbers may introduce large numerical
      errors in the algorithm.

NOTE: minlpsetbc() can  be  used  to  specify  different  constraints  for
      different variables.

NOTE: BndL>BndU will result in LP problem being recognized as infeasible.

  -- ALGLIB --
     Copyright 19.07.2018 by Bochkanov Sergey
*************************************************************************/
void minlpsetbci(const minlpstate &state, const ae_int_t i, const double bndl, const double bndu, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets one-sided linear constraints A*x ~ AU, where "~" can be
a mix of "<=", "=" and ">=".

IMPORTANT: this function is provided here for compatibility with the  rest
           of ALGLIB optimizers which accept constraints  in  format  like
           this one. Many real-life problems feature two-sided constraints
           like a0 <= a*x <= a1. It is really inefficient to add them as a
           pair of one-sided constraints.

           Use minlpsetlc2dense(), minlpsetlc2(), minlpaddlc2()  (or   its
           sparse version) wherever possible.

INPUT PARAMETERS:
    State   -   structure previously allocated with minlpcreate() call.
    A       -   linear constraints, array[K,N+1]. Each row of A represents
                one constraint, with first N elements being linear coefficients,
                and last element being right side.
    CT      -   constraint types, array[K]:
                * if CT[i]>0, then I-th constraint is A[i,*]*x >= A[i,n]
                * if CT[i]=0, then I-th constraint is A[i,*]*x  = A[i,n]
                * if CT[i]<0, then I-th constraint is A[i,*]*x <= A[i,n]
    K       -   number of equality/inequality constraints,  K>=0;  if  not
                given, inferred from sizes of A and CT.

  -- ALGLIB --
     Copyright 19.07.2018 by Bochkanov Sergey
*************************************************************************/
void minlpsetlc(const minlpstate &state, const real_2d_array &a, const integer_1d_array &ct, const ae_int_t k, const xparams _xparams = alglib::xdefault);
void minlpsetlc(const minlpstate &state, const real_2d_array &a, const integer_1d_array &ct, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets two-sided linear constraints AL <= A*x <= AU.

This version accepts dense matrix as  input;  internally  LP  solver  uses
sparse storage  anyway  (most  LP  problems  are  sparse),  but  for  your
convenience it may accept dense inputs. This  function  overwrites  linear
constraints set by previous calls (if such calls were made).

We recommend you to use sparse version of this function unless  you  solve
small-scale LP problem (less than few hundreds of variables).

NOTE: there also exist several versions of this function:
      * one-sided dense version which  accepts  constraints  in  the  same
        format as one used by QP and  NLP solvers
      * two-sided sparse version which accepts sparse matrix
      * two-sided dense  version which allows you to add constraints row by row
      * two-sided sparse version which allows you to add constraints row by row

INPUT PARAMETERS:
    State   -   structure previously allocated with minlpcreate() call.
    A       -   linear constraints, array[K,N]. Each row of  A  represents
                one  constraint. One-sided  inequality   constraints, two-
                sided inequality  constraints,  equality  constraints  are
                supported (see below)
    AL, AU  -   lower and upper bounds, array[K];
                * AL[i]=AU[i] => equality constraint Ai*x
                * AL[i]<AU[i] => two-sided constraint AL[i]<=Ai*x<=AU[i]
                * AL[i]=-INF  => one-sided constraint Ai*x<=AU[i]
                * AU[i]=+INF  => one-sided constraint AL[i]<=Ai*x
                * AL[i]=-INF, AU[i]=+INF => constraint is ignored
    K       -   number of equality/inequality constraints,  K>=0;  if  not
                given, inferred from sizes of A, AL, AU.

  -- ALGLIB --
     Copyright 19.07.2018 by Bochkanov Sergey
*************************************************************************/
void minlpsetlc2dense(const minlpstate &state, const real_2d_array &a, const real_1d_array &al, const real_1d_array &au, const ae_int_t k, const xparams _xparams = alglib::xdefault);
void minlpsetlc2dense(const minlpstate &state, const real_2d_array &a, const real_1d_array &al, const real_1d_array &au, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This  function  sets  two-sided linear  constraints  AL <= A*x <= AU  with
sparse constraining matrix A. Recommended for large-scale problems.

This  function  overwrites  linear  (non-box)  constraints set by previous
calls (if such calls were made).

INPUT PARAMETERS:
    State   -   structure previously allocated with minlpcreate() call.
    A       -   sparse matrix with size [K,N] (exactly!).
                Each row of A represents one general linear constraint.
                A can be stored in any sparse storage format.
    AL, AU  -   lower and upper bounds, array[K];
                * AL[i]=AU[i] => equality constraint Ai*x
                * AL[i]<AU[i] => two-sided constraint AL[i]<=Ai*x<=AU[i]
                * AL[i]=-INF  => one-sided constraint Ai*x<=AU[i]
                * AU[i]=+INF  => one-sided constraint AL[i]<=Ai*x
                * AL[i]=-INF, AU[i]=+INF => constraint is ignored
    K       -   number  of equality/inequality constraints, K>=0.  If  K=0
                is specified, A, AL, AU are ignored.

  -- ALGLIB --
     Copyright 19.07.2018 by Bochkanov Sergey
*************************************************************************/
void minlpsetlc2(const minlpstate &state, const sparsematrix &a, const real_1d_array &al, const real_1d_array &au, const ae_int_t k, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function appends two-sided linear constraint  AL <= A*x <= AU  to the
list of currently present constraints.

This version accepts dense constraint vector as input, but  sparsifies  it
for internal storage and processing. Thus, time to add one  constraint  in
is O(N) - we have to scan entire array of length N. Sparse version of this
function is order of magnitude faster for  constraints  with  just  a  few
nonzeros per row.

INPUT PARAMETERS:
    State   -   structure previously allocated with minlpcreate() call.
    A       -   linear constraint coefficient, array[N], right side is NOT
                included.
    AL, AU  -   lower and upper bounds;
                * AL=AU    => equality constraint Ai*x
                * AL<AU    => two-sided constraint AL<=A*x<=AU
                * AL=-INF  => one-sided constraint Ai*x<=AU
                * AU=+INF  => one-sided constraint AL<=Ai*x
                * AL=-INF, AU=+INF => constraint is ignored

  -- ALGLIB --
     Copyright 19.07.2018 by Bochkanov Sergey
*************************************************************************/
void minlpaddlc2dense(const minlpstate &state, const real_1d_array &a, const double al, const double au, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function appends two-sided linear constraint  AL <= A*x <= AU  to the
list of currently present constraints.

Constraint is passed in compressed format: as list of non-zero entries  of
coefficient vector A. Such approach is more efficient than  dense  storage
for highly sparse constraint vectors.

INPUT PARAMETERS:
    State   -   structure previously allocated with minlpcreate() call.
    IdxA    -   array[NNZ], indexes of non-zero elements of A:
                * can be unsorted
                * can include duplicate indexes (corresponding entries  of
                  ValA[] will be summed)
    ValA    -   array[NNZ], values of non-zero elements of A
    NNZ     -   number of non-zero coefficients in A
    AL, AU  -   lower and upper bounds;
                * AL=AU    => equality constraint A*x
                * AL<AU    => two-sided constraint AL<=A*x<=AU
                * AL=-INF  => one-sided constraint A*x<=AU
                * AU=+INF  => one-sided constraint AL<=A*x
                * AL=-INF, AU=+INF => constraint is ignored

  -- ALGLIB --
     Copyright 19.07.2018 by Bochkanov Sergey
*************************************************************************/
void minlpaddlc2(const minlpstate &state, const integer_1d_array &idxa, const real_1d_array &vala, const ae_int_t nnz, const double al, const double au, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function solves LP problem.

INPUT PARAMETERS:
    State   -   algorithm state

You should use minlpresults() function to access results  after  calls  to
this function.

  -- ALGLIB --
     Copyright 19.07.2018 by Bochkanov Sergey.
*************************************************************************/
void minlpoptimize(const minlpstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
LP solver results

INPUT PARAMETERS:
    State   -   algorithm state

OUTPUT PARAMETERS:
    X       -   array[N], solution. Filled by zeros on failure.
    Rep     -   optimization report. You should check Rep.TerminationType,
                which contains completion code, and you may check  another
                fields which contain another information  about  algorithm
                functioning.

                Failure codes returned by algorithm are:
                * -4    LP problem is primal unbounded (dual infeasible)
                * -3    LP problem is primal infeasible (dual unbounded)

                Success codes:
                *  1..4 successful completion
                *  5    MaxIts steps was taken

  -- ALGLIB --
     Copyright 11.01.2011 by Bochkanov Sergey
*************************************************************************/
void minlpresults(const minlpstate &state, real_1d_array &x, minlpreport &rep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
LP results

Buffered implementation of MinLPResults() which uses pre-allocated  buffer
to store X[]. If buffer size is  too  small,  it  resizes  buffer.  It  is
intended to be used in the inner cycles of performance critical algorithms
where array reallocation penalty is too large to be ignored.

  -- ALGLIB --
     Copyright 11.01.2011 by Bochkanov Sergey
*************************************************************************/
void minlpresultsbuf(const minlpstate &state, real_1d_array &x, minlpreport &rep, const xparams _xparams = alglib::xdefault);
#endif

#if defined(AE_COMPILE_NLCSLP) || !defined(AE_PARTIAL_BUILD)

#endif

#if defined(AE_COMPILE_MINNLC) || !defined(AE_PARTIAL_BUILD)
/*************************************************************************
                  NONLINEARLY  CONSTRAINED  OPTIMIZATION
            WITH PRECONDITIONED AUGMENTED LAGRANGIAN ALGORITHM

DESCRIPTION:
The  subroutine  minimizes  function   F(x)  of N arguments subject to any
combination of:
* bound constraints
* linear inequality constraints
* linear equality constraints
* nonlinear equality constraints Gi(x)=0
* nonlinear inequality constraints Hi(x)<=0

REQUIREMENTS:
* user must provide function value and gradient for F(), H(), G()
* starting point X0 must be feasible or not too far away from the feasible
  set
* F(), G(), H() are twice continuously differentiable on the feasible  set
  and its neighborhood
* nonlinear constraints G() and H() must have non-zero gradient at  G(x)=0
  and at H(x)=0. Say, constraint like x^2>=1 is supported, but x^2>=0   is
  NOT supported.

USAGE:

Constrained optimization if far more complex than the  unconstrained  one.
Nonlinearly constrained optimization is one of the most esoteric numerical
procedures.

Here we give very brief outline  of  the  MinNLC  optimizer.  We  strongly
recommend you to study examples in the ALGLIB Reference Manual and to read
ALGLIB User Guide on optimization, which is available at
http://www.alglib.net/optimization/

1. User initializes algorithm state with MinNLCCreate() call  and  chooses
   what NLC solver to use. There is some solver which is used by  default,
   with default settings, but you should NOT rely on  default  choice.  It
   may change in future releases of ALGLIB without notice, and no one  can
   guarantee that new solver will be  able  to  solve  your  problem  with
   default settings.

   From the other side, if you choose solver explicitly, you can be pretty
   sure that it will work with new ALGLIB releases.

   In the current release following solvers can be used:
   * SLP solver (activated with minnlcsetalgoslp() function) -  successive
     linear programming, recommended as the first step.
   * AUL solver (activated with minnlcsetalgoaul() function)  -  augmented
     Lagrangian method with dense preconditioner.

   SLP solver is the most robust one  in  ALGLIB  and  converges  in  less
   iterations than AUL; however, each iteration has higher overhead  -  we
   have to solve an LP problem. From  the  other  side,  AUL  has  cheaper
   iterations - although it typically needs more of them, and also  it  is
   less robust in nonconvex setting.

2. [optional] user activates OptGuard  integrity checker  which  tries  to
   detect possible errors in the user-supplied callbacks:
   * discontinuity/nonsmoothness of the target/nonlinear constraints
   * errors in the analytic gradient provided by user
   This feature is essential for early prototyping stages because it helps
   to catch common coding and problem statement errors.
   OptGuard can be activated with following functions (one per each  check
   performed):
   * minnlcoptguardsmoothness()
   * minnlcoptguardgradient()

3. User adds boundary and/or linear and/or nonlinear constraints by  means
   of calling one of the following functions:
   a) minnlcsetbc() for boundary constraints
   b) minnlcsetlc() for linear constraints
   c) minnlcsetnlc() for nonlinear constraints
   You may combine (a), (b) and (c) in one optimization problem.

4. User sets scale of the variables with minnlcsetscale() function. It  is
   VERY important to set  scale  of  the  variables,  because  nonlinearly
   constrained problems are hard to solve when variables are badly scaled.

5. User sets  stopping  conditions  with  minnlcsetcond(). If  NLC  solver
   uses  inner/outer  iteration  layout,  this  function   sets   stopping
   conditions for INNER iterations.

6. Finally, user calls minnlcoptimize()  function  which  takes  algorithm
   state and pointer (delegate, etc.) to callback function which calculates
   F/G/H.

7. User calls  minnlcresults()  to  get  solution;  additionally  you  can
   retrieve OptGuard report with minnlcoptguardresults(), and get detailed
   report about purported errors in the target function with:
   * minnlcoptguardnonc1test0results()
   * minnlcoptguardnonc1test1results()

8. Optionally user may call minnlcrestartfrom() to solve  another  problem
   with same N but another starting point. minnlcrestartfrom()  allows  to
   reuse already initialized structure.


INPUT PARAMETERS:
    N       -   problem dimension, N>0:
                * if given, only leading N elements of X are used
                * if not given, automatically determined from size ofX
    X       -   starting point, array[N]:
                * it is better to set X to a feasible point
                * but X can be infeasible, in which case algorithm will try
                  to find feasible point first, using X as initial
                  approximation.

OUTPUT PARAMETERS:
    State   -   structure stores algorithm state

  -- ALGLIB --
     Copyright 06.06.2014 by Bochkanov Sergey
*************************************************************************/
void minnlccreate(const ae_int_t n, const real_1d_array &x, minnlcstate &state, const xparams _xparams = alglib::xdefault);
void minnlccreate(const real_1d_array &x, minnlcstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This subroutine is a finite  difference variant of MinNLCCreate(). It uses
finite differences in order to differentiate target function.

Description below contains information which is specific to this  function
only. We recommend to read comments on MinNLCCreate() in order to get more
information about creation of NLC optimizer.

INPUT PARAMETERS:
    N       -   problem dimension, N>0:
                * if given, only leading N elements of X are used
                * if not given, automatically determined from size ofX
    X       -   starting point, array[N]:
                * it is better to set X to a feasible point
                * but X can be infeasible, in which case algorithm will try
                  to find feasible point first, using X as initial
                  approximation.
    DiffStep-   differentiation step, >0

OUTPUT PARAMETERS:
    State   -   structure stores algorithm state

NOTES:
1. algorithm uses 4-point central formula for differentiation.
2. differentiation step along I-th axis is equal to DiffStep*S[I] where
   S[] is scaling vector which can be set by MinNLCSetScale() call.
3. we recommend you to use moderate values of  differentiation  step.  Too
   large step will result in too large TRUNCATION  errors, while too small
   step will result in too large NUMERICAL  errors.  1.0E-4  can  be  good
   value to start from.
4. Numerical  differentiation  is   very   inefficient  -   one   gradient
   calculation needs 4*N function evaluations. This function will work for
   any N - either small (1...10), moderate (10...100) or  large  (100...).
   However, performance penalty will be too severe for any N's except  for
   small ones.
   We should also say that code which relies on numerical  differentiation
   is  less   robust   and  precise.  Imprecise  gradient  may  slow  down
   convergence, especially on highly nonlinear problems.
   Thus  we  recommend to use this function for fast prototyping on small-
   dimensional problems only, and to implement analytical gradient as soon
   as possible.

  -- ALGLIB --
     Copyright 06.06.2014 by Bochkanov Sergey
*************************************************************************/
void minnlccreatef(const ae_int_t n, const real_1d_array &x, const double diffstep, minnlcstate &state, const xparams _xparams = alglib::xdefault);
void minnlccreatef(const real_1d_array &x, const double diffstep, minnlcstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets boundary constraints for NLC optimizer.

Boundary constraints are inactive by  default  (after  initial  creation).
They are preserved after algorithm restart with  MinNLCRestartFrom().

You may combine boundary constraints with  general  linear ones - and with
nonlinear ones! Boundary constraints are  handled  more  efficiently  than
other types.  Thus,  if  your  problem  has  mixed  constraints,  you  may
explicitly specify some of them as boundary and save some time/space.

INPUT PARAMETERS:
    State   -   structure stores algorithm state
    BndL    -   lower bounds, array[N].
                If some (all) variables are unbounded, you may specify
                very small number or -INF.
    BndU    -   upper bounds, array[N].
                If some (all) variables are unbounded, you may specify
                very large number or +INF.

NOTE 1:  it is possible to specify  BndL[i]=BndU[i].  In  this  case  I-th
variable will be "frozen" at X[i]=BndL[i]=BndU[i].

NOTE 2:  when you solve your problem  with  augmented  Lagrangian  solver,
         boundary constraints are  satisfied  only  approximately!  It  is
         possible   that  algorithm  will  evaluate  function  outside  of
         feasible area!

  -- ALGLIB --
     Copyright 06.06.2014 by Bochkanov Sergey
*************************************************************************/
void minnlcsetbc(const minnlcstate &state, const real_1d_array &bndl, const real_1d_array &bndu, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets linear constraints for MinNLC optimizer.

Linear constraints are inactive by default (after initial creation).  They
are preserved after algorithm restart with MinNLCRestartFrom().

You may combine linear constraints with boundary ones - and with nonlinear
ones! If your problem has mixed constraints, you  may  explicitly  specify
some of them as linear. It  may  help  optimizer   to   handle  them  more
efficiently.

INPUT PARAMETERS:
    State   -   structure previously allocated with MinNLCCreate call.
    C       -   linear constraints, array[K,N+1].
                Each row of C represents one constraint, either equality
                or inequality (see below):
                * first N elements correspond to coefficients,
                * last element corresponds to the right part.
                All elements of C (including right part) must be finite.
    CT      -   type of constraints, array[K]:
                * if CT[i]>0, then I-th constraint is C[i,*]*x >= C[i,n+1]
                * if CT[i]=0, then I-th constraint is C[i,*]*x  = C[i,n+1]
                * if CT[i]<0, then I-th constraint is C[i,*]*x <= C[i,n+1]
    K       -   number of equality/inequality constraints, K>=0:
                * if given, only leading K elements of C/CT are used
                * if not given, automatically determined from sizes of C/CT

NOTE 1: when you solve your problem  with  augmented  Lagrangian   solver,
        linear constraints are  satisfied  only   approximately!   It   is
        possible   that  algorithm  will  evaluate  function  outside   of
        feasible area!

  -- ALGLIB --
     Copyright 06.06.2014 by Bochkanov Sergey
*************************************************************************/
void minnlcsetlc(const minnlcstate &state, const real_2d_array &c, const integer_1d_array &ct, const ae_int_t k, const xparams _xparams = alglib::xdefault);
void minnlcsetlc(const minnlcstate &state, const real_2d_array &c, const integer_1d_array &ct, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets nonlinear constraints for MinNLC optimizer.

In fact, this function sets NUMBER of nonlinear  constraints.  Constraints
itself (constraint functions) are passed to MinNLCOptimize() method.  This
method requires user-defined vector function F[]  and  its  Jacobian  J[],
where:
* first component of F[] and first row  of  Jacobian  J[]  corresponds  to
  function being minimized
* next NLEC components of F[] (and rows  of  J)  correspond  to  nonlinear
  equality constraints G_i(x)=0
* next NLIC components of F[] (and rows  of  J)  correspond  to  nonlinear
  inequality constraints H_i(x)<=0

NOTE: you may combine nonlinear constraints with linear/boundary ones.  If
      your problem has mixed constraints, you  may explicitly specify some
      of them as linear ones. It may help optimizer to  handle  them  more
      efficiently.

INPUT PARAMETERS:
    State   -   structure previously allocated with MinNLCCreate call.
    NLEC    -   number of Non-Linear Equality Constraints (NLEC), >=0
    NLIC    -   number of Non-Linear Inquality Constraints (NLIC), >=0

NOTE 1: when you solve your problem  with  augmented  Lagrangian   solver,
        nonlinear constraints are satisfied only  approximately!   It   is
        possible   that  algorithm  will  evaluate  function  outside   of
        feasible area!

NOTE 2: algorithm scales variables  according  to   scale   specified   by
        MinNLCSetScale()  function,  so  it can handle problems with badly
        scaled variables (as long as we KNOW their scales).

        However,  there  is  no  way  to  automatically  scale   nonlinear
        constraints Gi(x) and Hi(x). Inappropriate scaling  of  Gi/Hi  may
        ruin convergence. Solving problem with  constraint  "1000*G0(x)=0"
        is NOT same as solving it with constraint "0.001*G0(x)=0".

        It  means  that  YOU  are  the  one who is responsible for correct
        scaling of nonlinear constraints Gi(x) and Hi(x). We recommend you
        to scale nonlinear constraints in such way that I-th component  of
        dG/dX (or dH/dx) has approximately unit  magnitude  (for  problems
        with unit scale)  or  has  magnitude approximately equal to 1/S[i]
        (where S is a scale set by MinNLCSetScale() function).


  -- ALGLIB --
     Copyright 06.06.2014 by Bochkanov Sergey
*************************************************************************/
void minnlcsetnlc(const minnlcstate &state, const ae_int_t nlec, const ae_int_t nlic, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets stopping conditions for inner iterations of  optimizer.

INPUT PARAMETERS:
    State   -   structure which stores algorithm state
    EpsX    -   >=0
                The subroutine finishes its work if  on  k+1-th  iteration
                the condition |v|<=EpsX is fulfilled, where:
                * |.| means Euclidian norm
                * v - scaled step vector, v[i]=dx[i]/s[i]
                * dx - step vector, dx=X(k+1)-X(k)
                * s - scaling coefficients set by MinNLCSetScale()
    MaxIts  -   maximum number of iterations. If MaxIts=0, the  number  of
                iterations is unlimited.

Passing EpsX=0 and MaxIts=0 (simultaneously) will lead to automatic
selection of the stopping condition.

  -- ALGLIB --
     Copyright 06.06.2014 by Bochkanov Sergey
*************************************************************************/
void minnlcsetcond(const minnlcstate &state, const double epsx, const ae_int_t maxits, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets scaling coefficients for NLC optimizer.

ALGLIB optimizers use scaling matrices to test stopping  conditions  (step
size and gradient are scaled before comparison with tolerances).  Scale of
the I-th variable is a translation invariant measure of:
a) "how large" the variable is
b) how large the step should be to make significant changes in the function

Scaling is also used by finite difference variant of the optimizer  - step
along I-th axis is equal to DiffStep*S[I].

INPUT PARAMETERS:
    State   -   structure stores algorithm state
    S       -   array[N], non-zero scaling coefficients
                S[i] may be negative, sign doesn't matter.

  -- ALGLIB --
     Copyright 06.06.2014 by Bochkanov Sergey
*************************************************************************/
void minnlcsetscale(const minnlcstate &state, const real_1d_array &s, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets preconditioner to "inexact LBFGS-based" mode.

Preconditioning is very important for convergence of  Augmented Lagrangian
algorithm because presence of penalty term makes problem  ill-conditioned.
Difference between  performance  of  preconditioned  and  unpreconditioned
methods can be as large as 100x!

MinNLC optimizer may use following preconditioners,  each  with   its  own
benefits and drawbacks:
    a) inexact LBFGS-based, with O(N*K) evaluation time
    b) exact low rank one,  with O(N*K^2) evaluation time
    c) exact robust one,    with O(N^3+K*N^2) evaluation time
where K is a total number of general linear and nonlinear constraints (box
ones are not counted).

Inexact  LBFGS-based  preconditioner  uses L-BFGS  formula  combined  with
orthogonality assumption to perform very fast updates. For a N-dimensional
problem with K general linear or nonlinear constraints (boundary ones  are
not counted) it has O(N*K) cost per iteration.  This   preconditioner  has
best  quality  (less  iterations)  when   general   linear  and  nonlinear
constraints are orthogonal to each other (orthogonality  with  respect  to
boundary constraints is not required). Number of iterations increases when
constraints  are  non-orthogonal, because algorithm assumes orthogonality,
but still it is better than no preconditioner at all.

INPUT PARAMETERS:
    State   -   structure stores algorithm state

  -- ALGLIB --
     Copyright 26.09.2014 by Bochkanov Sergey
*************************************************************************/
void minnlcsetprecinexact(const minnlcstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets preconditioner to "exact low rank" mode.

Preconditioning is very important for convergence of  Augmented Lagrangian
algorithm because presence of penalty term makes problem  ill-conditioned.
Difference between  performance  of  preconditioned  and  unpreconditioned
methods can be as large as 100x!

MinNLC optimizer may use following preconditioners,  each  with   its  own
benefits and drawbacks:
    a) inexact LBFGS-based, with O(N*K) evaluation time
    b) exact low rank one,  with O(N*K^2) evaluation time
    c) exact robust one,    with O(N^3+K*N^2) evaluation time
where K is a total number of general linear and nonlinear constraints (box
ones are not counted).

It also provides special unpreconditioned mode of operation which  can  be
used for test purposes. Comments below discuss low rank preconditioner.

Exact low-rank preconditioner  uses  Woodbury  matrix  identity  to  build
quadratic model of the penalized function. It has following features:
* no special assumptions about orthogonality of constraints
* preconditioner evaluation is optimized for K<<N. Its cost  is  O(N*K^2),
  so it may become prohibitively slow for K>=N.
* finally, stability of the process is guaranteed only for K<<N.  Woodbury
  update often fail for K>=N due to degeneracy of  intermediate  matrices.
  That's why we recommend to use "exact robust"  preconditioner  for  such
  cases.

RECOMMENDATIONS

We  recommend  to  choose  between  "exact  low  rank"  and "exact robust"
preconditioners, with "low rank" version being chosen  when  you  know  in
advance that total count of non-box constraints won't exceed N, and "robust"
version being chosen when you need bulletproof solution.

INPUT PARAMETERS:
    State   -   structure stores algorithm state
    UpdateFreq- update frequency. Preconditioner is  rebuilt  after  every
                UpdateFreq iterations. Recommended value: 10 or higher.
                Zero value means that good default value will be used.

  -- ALGLIB --
     Copyright 26.09.2014 by Bochkanov Sergey
*************************************************************************/
void minnlcsetprecexactlowrank(const minnlcstate &state, const ae_int_t updatefreq, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets preconditioner to "exact robust" mode.

Preconditioning is very important for convergence of  Augmented Lagrangian
algorithm because presence of penalty term makes problem  ill-conditioned.
Difference between  performance  of  preconditioned  and  unpreconditioned
methods can be as large as 100x!

MinNLC optimizer may use following preconditioners,  each  with   its  own
benefits and drawbacks:
    a) inexact LBFGS-based, with O(N*K) evaluation time
    b) exact low rank one,  with O(N*K^2) evaluation time
    c) exact robust one,    with O(N^3+K*N^2) evaluation time
where K is a total number of general linear and nonlinear constraints (box
ones are not counted).

It also provides special unpreconditioned mode of operation which  can  be
used for test purposes. Comments below discuss robust preconditioner.

Exact  robust  preconditioner   uses   Cholesky  decomposition  to  invert
approximate Hessian matrix H=D+W'*C*W (where D stands for  diagonal  terms
of Hessian, combined result of initial scaling matrix and penalty from box
constraints; W stands for general linear constraints and linearization  of
nonlinear ones; C stands for diagonal matrix of penalty coefficients).

This preconditioner has following features:
* no special assumptions about constraint structure
* preconditioner is optimized  for  stability;  unlike  "exact  low  rank"
  version which fails for K>=N, this one works well for any value of K.
* the only drawback is that is takes O(N^3+K*N^2) time  to  build  it.  No
  economical  Woodbury update is applied even when it  makes  sense,  thus
  there  are  exist situations (K<<N) when "exact low rank" preconditioner
  outperforms this one.

RECOMMENDATIONS

We  recommend  to  choose  between  "exact  low  rank"  and "exact robust"
preconditioners, with "low rank" version being chosen  when  you  know  in
advance that total count of non-box constraints won't exceed N, and "robust"
version being chosen when you need bulletproof solution.

INPUT PARAMETERS:
    State   -   structure stores algorithm state
    UpdateFreq- update frequency. Preconditioner is  rebuilt  after  every
                UpdateFreq iterations. Recommended value: 10 or higher.
                Zero value means that good default value will be used.

  -- ALGLIB --
     Copyright 26.09.2014 by Bochkanov Sergey
*************************************************************************/
void minnlcsetprecexactrobust(const minnlcstate &state, const ae_int_t updatefreq, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets preconditioner to "turned off" mode.

Preconditioning is very important for convergence of  Augmented Lagrangian
algorithm because presence of penalty term makes problem  ill-conditioned.
Difference between  performance  of  preconditioned  and  unpreconditioned
methods can be as large as 100x!

MinNLC optimizer may  utilize  two  preconditioners,  each  with  its  own
benefits and drawbacks: a) inexact LBFGS-based, and b) exact low rank one.
It also provides special unpreconditioned mode of operation which  can  be
used for test purposes.

This function activates this test mode. Do not use it in  production  code
to solve real-life problems.

INPUT PARAMETERS:
    State   -   structure stores algorithm state

  -- ALGLIB --
     Copyright 26.09.2014 by Bochkanov Sergey
*************************************************************************/
void minnlcsetprecnone(const minnlcstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets maximum step length (after scaling of step vector  with
respect to variable scales specified by minnlcsetscale() call).

INPUT PARAMETERS:
    State   -   structure which stores algorithm state
    StpMax  -   maximum step length, >=0. Set StpMax to 0.0 (default),  if
                you don't want to limit step length.

Use this subroutine when you optimize target function which contains exp()
or  other  fast  growing  functions,  and optimization algorithm makes too
large  steps  which  leads  to overflow. This function allows us to reject
steps  that  are  too  large  (and  therefore  expose  us  to the possible
overflow) without actually calculating function value at the x+stp*d.

NOTE: different solvers employed by MinNLC optimizer use  different  norms
      for step; AUL solver uses 2-norm, whilst SLP solver uses INF-norm.

  -- ALGLIB --
     Copyright 02.04.2010 by Bochkanov Sergey
*************************************************************************/
void minnlcsetstpmax(const minnlcstate &state, const double stpmax, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This  function  tells MinNLC unit to use  Augmented  Lagrangian  algorithm
for nonlinearly constrained  optimization.  This  algorithm  is  a  slight
modification of one described in "A Modified Barrier-Augmented  Lagrangian
Method for  Constrained  Minimization  (1999)"  by  D.GOLDFARB,  R.POLYAK,
K. SCHEINBERG, I.YUZEFOVICH.

AUL solver can be significantly faster than SLP on easy problems, although
it is less robust than SLP (the "gold standard" of robust optimization).

Augmented Lagrangian algorithm works by converting problem  of  minimizing
F(x) subject to equality/inequality constraints   to unconstrained problem
of the form

    min[ f(x) +
        + Rho*PENALTY_EQ(x)   + SHIFT_EQ(x,Nu1) +
        + Rho*PENALTY_INEQ(x) + SHIFT_INEQ(x,Nu2) ]

where:
* Rho is a fixed penalization coefficient
* PENALTY_EQ(x) is a penalty term, which is used to APPROXIMATELY  enforce
  equality constraints
* SHIFT_EQ(x) is a special "shift"  term  which  is  used  to  "fine-tune"
  equality constraints, greatly increasing precision
* PENALTY_INEQ(x) is a penalty term which is used to approximately enforce
  inequality constraints
* SHIFT_INEQ(x) is a special "shift"  term  which  is  used to "fine-tune"
  inequality constraints, greatly increasing precision
* Nu1/Nu2 are vectors of Lagrange coefficients which are fine-tuned during
  outer iterations of algorithm

This  version  of  AUL  algorithm  uses   preconditioner,  which   greatly
accelerates convergence. Because this  algorithm  is  similar  to  penalty
methods,  it  may  perform  steps  into  infeasible  area.  All  kinds  of
constraints (boundary, linear and nonlinear ones) may   be   violated   in
intermediate points - and in the solution.  However,  properly  configured
AUL method is significantly better at handling  constraints  than  barrier
and/or penalty methods.

The very basic outline of algorithm is given below:
1) first outer iteration is performed with "default"  values  of  Lagrange
   multipliers Nu1/Nu2. Solution quality is low (candidate  point  can  be
   too  far  away  from  true  solution; large violation of constraints is
   possible) and is comparable with that of penalty methods.
2) subsequent outer iterations  refine  Lagrange  multipliers  and improve
   quality of the solution.

INPUT PARAMETERS:
    State   -   structure which stores algorithm state
    Rho     -   penalty coefficient, Rho>0:
                * large enough  that  algorithm  converges  with   desired
                  precision. Minimum value is 10*max(S'*diag(H)*S),  where
                  S is a scale matrix (set by MinNLCSetScale) and H  is  a
                  Hessian of the function being minimized. If you can  not
                  easily estimate Hessian norm,  see  our  recommendations
                  below.
                * not TOO large to prevent ill-conditioning
                * for unit-scale problems (variables and Hessian have unit
                  magnitude), Rho=100 or Rho=1000 can be used.
                * it is important to note that Rho is internally multiplied
                  by scaling matrix, i.e. optimum value of Rho depends  on
                  scale of variables specified  by  MinNLCSetScale().
    ItsCnt  -   number of outer iterations:
                * ItsCnt=0 means that small number of outer iterations  is
                  automatically chosen (10 iterations in current version).
                * ItsCnt=1 means that AUL algorithm performs just as usual
                  barrier method.
                * ItsCnt>1 means that  AUL  algorithm  performs  specified
                  number of outer iterations

HOW TO CHOOSE PARAMETERS

Nonlinear optimization is a tricky area and Augmented Lagrangian algorithm
is sometimes hard to tune. Good values of  Rho  and  ItsCnt  are  problem-
specific.  In  order  to  help  you   we   prepared   following   set   of
recommendations:

* for  unit-scale  problems  (variables  and Hessian have unit magnitude),
  Rho=100 or Rho=1000 can be used.

* start from  some  small  value of Rho and solve problem  with  just  one
  outer iteration (ItcCnt=1). In this case algorithm behaves like  penalty
  method. Increase Rho in 2x or 10x steps until you  see  that  one  outer
  iteration returns point which is "rough approximation to solution".

  It is very important to have Rho so  large  that  penalty  term  becomes
  constraining i.e. modified function becomes highly convex in constrained
  directions.

  From the other side, too large Rho may prevent you  from  converging  to
  the solution. You can diagnose it by studying number of inner iterations
  performed by algorithm: too few (5-10 on  1000-dimensional  problem)  or
  too many (orders of magnitude more than  dimensionality)  usually  means
  that Rho is too large.

* with just one outer iteration you  usually  have  low-quality  solution.
  Some constraints can be violated with very  large  margin,  while  other
  ones (which are NOT violated in the true solution) can push final  point
  too far in the inner area of the feasible set.

  For example, if you have constraint x0>=0 and true solution  x0=1,  then
  merely a presence of "x0>=0" will introduce a bias towards larger values
  of x0. Say, algorithm may stop at x0=1.5 instead of 1.0.

* after you found good Rho, you may increase number of  outer  iterations.
  ItsCnt=10 is a good value. Subsequent outer iteration will refine values
  of  Lagrange  multipliers.  Constraints  which  were  violated  will  be
  enforced, inactive constraints will be dropped (corresponding multipliers
  will be decreased). Ideally, you  should  see  10-1000x  improvement  in
  constraint handling (constraint violation is reduced).

* if  you  see  that  algorithm  converges  to  vicinity  of solution, but
  additional outer iterations do not refine solution,  it  may  mean  that
  algorithm is unstable - it wanders around true  solution,  but  can  not
  approach it. Sometimes algorithm may be stabilized by increasing Rho one
  more time, making it 5x or 10x larger.

SCALING OF CONSTRAINTS [IMPORTANT]

AUL optimizer scales   variables   according   to   scale   specified   by
MinNLCSetScale() function, so it can handle  problems  with  badly  scaled
variables (as long as we KNOW their scales).   However,  because  function
being optimized is a mix  of  original  function and  constraint-dependent
penalty  functions, it  is   important  to   rescale  both  variables  AND
constraints.

Say,  if  you  minimize f(x)=x^2 subject to 1000000*x>=0,  then  you  have
constraint whose scale is different from that of target  function (another
example is 0.000001*x>=0). It is also possible to have constraints   whose
scales  are   misaligned:   1000000*x0>=0, 0.000001*x1<=0.   Inappropriate
scaling may ruin convergence because minimizing x^2 subject to x>=0 is NOT
same as minimizing it subject to 1000000*x>=0.

Because we  know  coefficients  of  boundary/linear  constraints,  we  can
automatically rescale and normalize them. However,  there  is  no  way  to
automatically rescale nonlinear constraints Gi(x) and  Hi(x)  -  they  are
black boxes.

It means that YOU are the one who is  responsible  for  correct scaling of
nonlinear constraints  Gi(x)  and  Hi(x).  We  recommend  you  to  rescale
nonlinear constraints in such way that I-th component of dG/dX (or  dH/dx)
has magnitude approximately equal to 1/S[i] (where S  is  a  scale  set by
MinNLCSetScale() function).

WHAT IF IT DOES NOT CONVERGE?

It is possible that AUL algorithm fails to converge to precise  values  of
Lagrange multipliers. It stops somewhere around true solution, but candidate
point is still too far from solution, and some constraints  are  violated.
Such kind of failure is specific for Lagrangian algorithms -  technically,
they stop at some point, but this point is not constrained solution.

There are exist several reasons why algorithm may fail to converge:
a) too loose stopping criteria for inner iteration
b) degenerate, redundant constraints
c) target function has unconstrained extremum exactly at the  boundary  of
   some constraint
d) numerical noise in the target function

In all these cases algorithm is unstable - each outer iteration results in
large and almost random step which improves handling of some  constraints,
but violates other ones (ideally  outer iterations should form a  sequence
of progressively decreasing steps towards solution).

First reason possible is  that  too  loose  stopping  criteria  for  inner
iteration were specified. Augmented Lagrangian algorithm solves a sequence
of intermediate problems, and requries each of them to be solved with high
precision. Insufficient precision results in incorrect update of  Lagrange
multipliers.

Another reason is that you may have specified degenerate constraints: say,
some constraint was repeated twice. In most cases AUL algorithm gracefully
handles such situations, but sometimes it may spend too much time figuring
out subtle degeneracies in constraint matrix.

Third reason is tricky and hard to diagnose. Consider situation  when  you
minimize  f=x^2  subject to constraint x>=0.  Unconstrained   extremum  is
located  exactly  at  the  boundary  of  constrained  area.  In  this case
algorithm will tend to oscillate between negative  and  positive  x.  Each
time it stops at x<0 it "reinforces" constraint x>=0, and each time it  is
bounced to x>0 it "relaxes" constraint (and is  attracted  to  x<0).

Such situation  sometimes  happens  in  problems  with  hidden  symetries.
Algorithm  is  got  caught  in  a  loop with  Lagrange  multipliers  being
continuously increased/decreased. Luckily, such loop forms after at  least
three iterations, so this problem can be solved by  DECREASING  number  of
outer iterations down to 1-2 and increasing  penalty  coefficient  Rho  as
much as possible.

Final reason is numerical noise. AUL algorithm is robust against  moderate
noise (more robust than, say, active set methods),  but  large  noise  may
destabilize algorithm.

  -- ALGLIB --
     Copyright 06.06.2014 by Bochkanov Sergey
*************************************************************************/
void minnlcsetalgoaul(const minnlcstate &state, const double rho, const ae_int_t itscnt, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This   function  tells  MinNLC  optimizer  to  use  SLP (Successive Linear
Programming) algorithm for  nonlinearly  constrained   optimization.  This
algorithm  is  a  slight  modification  of  one  described  in  "A  Linear
programming-based optimization algorithm for solving nonlinear programming
problems" (2010) by Claus Still and Tapio Westerlund.

Despite its name ("linear" = "first order method") this algorithm performs
steps similar to that of conjugate gradients method;  internally  it  uses
orthogonality/conjugacy requirement for subsequent steps  which  makes  it
closer to second order methods in terms of convergence speed.

Convergence is proved for the following case:
* function and constraints are continuously differentiable (C1 class)
* extended Mangasarian–Fromovitz constraint qualification  (EMFCQ)  holds;
  in the context of this algorithm EMFCQ  means  that  one  can,  for  any
  infeasible  point,  find  a  search  direction  such that the constraint
  infeasibilities are reduced.

This algorithm has following nice properties:
* no parameters to tune
* no convexity requirements for target function or constraints
* initial point can be infeasible
* algorithm respects box constraints in all intermediate points  (it  does
  not even evaluate function outside of box constrained area)
* once linear constraints are enforced, algorithm will not violate them
* no such guarantees can be provided for nonlinear constraints,  but  once
  nonlinear constraints are enforced, algorithm will try  to  respect them
  as much as possible
* numerical differentiation does not  violate  box  constraints  (although
  general linear and nonlinear ones can be violated during differentiation)

However, following drawbacks can be noted:
* algorithm performance decreased on problems with dense constraints
* it has higher iteration cost than AUL - we have to solve an  LP  problem
  at each step.

We recommend this algorithm as a first step; as soon as you make sure that
it converges, you can try switching to AUL which is sometimes much faster.

INPUT PARAMETERS:
    State   -   structure which stores algorithm state

  -- ALGLIB --
     Copyright 02.04.2018 by Bochkanov Sergey
*************************************************************************/
void minnlcsetalgoslp(const minnlcstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function turns on/off reporting.

INPUT PARAMETERS:
    State   -   structure which stores algorithm state
    NeedXRep-   whether iteration reports are needed or not

If NeedXRep is True, algorithm will call rep() callback function if  it is
provided to MinNLCOptimize().

NOTE: algorithm passes two parameters to rep() callback  -  current  point
      and penalized function value at current point. Important -  function
      value which is returned is NOT function being minimized. It  is  sum
      of the value of the function being minimized - and penalty term.

  -- ALGLIB --
     Copyright 28.11.2010 by Bochkanov Sergey
*************************************************************************/
void minnlcsetxrep(const minnlcstate &state, const bool needxrep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function provides reverse communication interface
Reverse communication interface is not documented or recommended to use.
See below for functions which provide better documented API
*************************************************************************/
bool minnlciteration(const minnlcstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This family of functions is used to launcn iterations of nonlinear optimizer

These functions accept following parameters:
    state   -   algorithm state
    fvec    -   callback which calculates function vector fi[]
                at given point x
    jac     -   callback which calculates function vector fi[]
                and Jacobian jac at given point x
    rep     -   optional callback which is called after each iteration
                can be NULL
    ptr     -   optional pointer which is passed to func/grad/hess/jac/rep
                can be NULL


NOTES:

1. This function has two different implementations: one which  uses  exact
   (analytical) user-supplied Jacobian, and one which uses  only  function
   vector and numerically  differentiates  function  in  order  to  obtain
   gradient.

   Depending  on  the  specific  function  used to create optimizer object
   you should choose appropriate variant of MinNLCOptimize() -  one  which
   accepts function AND Jacobian or one which accepts ONLY function.

   Be careful to choose variant of MinNLCOptimize()  which  corresponds to
   your optimization scheme! Table below lists different  combinations  of
   callback (function/gradient) passed to MinNLCOptimize()   and  specific
   function used to create optimizer.


                     |         USER PASSED TO MinNLCOptimize()
   CREATED WITH      |  function only   |  function and gradient
   ------------------------------------------------------------
   MinNLCCreateF()   |     works               FAILS
   MinNLCCreate()    |     FAILS               works

   Here "FAILS" denotes inappropriate combinations  of  optimizer creation
   function  and  MinNLCOptimize()  version.   Attemps   to    use    such
   combination will lead to exception. Either  you  did  not pass gradient
   when it WAS needed or you passed gradient when it was NOT needed.

  -- ALGLIB --
     Copyright 06.06.2014 by Bochkanov Sergey

*************************************************************************/
void minnlcoptimize(minnlcstate &state,
    void (*fvec)(const real_1d_array &x, real_1d_array &fi, void *ptr),
    void  (*rep)(const real_1d_array &x, double func, void *ptr) = NULL,
    void *ptr = NULL,
    const xparams _xparams = alglib::xdefault);
void minnlcoptimize(minnlcstate &state,
    void  (*jac)(const real_1d_array &x, real_1d_array &fi, real_2d_array &jac, void *ptr),
    void  (*rep)(const real_1d_array &x, double func, void *ptr) = NULL,
    void *ptr = NULL,
    const xparams _xparams = alglib::xdefault);


/*************************************************************************
This  function  activates/deactivates verification  of  the  user-supplied
analytic gradient/Jacobian.

Upon  activation  of  this  option  OptGuard  integrity  checker  performs
numerical differentiation of your target  function  (constraints)  at  the
initial point (note: future versions may also perform check  at  the final
point) and compares numerical gradient/Jacobian with analytic one provided
by you.

If difference is too large, an error flag is set and optimization  session
continues. After optimization session is over, you can retrieve the report
which stores both gradients/Jacobians, and specific components highlighted
as suspicious by the OptGuard.

The primary OptGuard report can be retrieved with minnlcoptguardresults().

IMPORTANT: gradient check is a high-overhead option which  will  cost  you
           about 3*N additional function evaluations. In many cases it may
           cost as much as the rest of the optimization session.

           YOU SHOULD NOT USE IT IN THE PRODUCTION CODE UNLESS YOU WANT TO
           CHECK DERIVATIVES PROVIDED BY SOME THIRD PARTY.

NOTE: unlike previous incarnation of the gradient checking code,  OptGuard
      does NOT interrupt optimization even if it discovers bad gradient.

INPUT PARAMETERS:
    State       -   structure used to store algorithm state
    TestStep    -   verification step used for numerical differentiation:
                    * TestStep=0 turns verification off
                    * TestStep>0 activates verification
                    You should carefully choose TestStep. Value  which  is
                    too large (so large that  function  behavior  is  non-
                    cubic at this scale) will lead  to  false  alarms. Too
                    short step will result in rounding  errors  dominating
                    numerical derivative.

                    You may use different step for different parameters by
                    means of setting scale with minnlcsetscale().

=== EXPLANATION ==========================================================

In order to verify gradient algorithm performs following steps:
  * two trial steps are made to X[i]-TestStep*S[i] and X[i]+TestStep*S[i],
    where X[i] is i-th component of the initial point and S[i] is a  scale
    of i-th parameter
  * F(X) is evaluated at these trial points
  * we perform one more evaluation in the middle point of the interval
  * we  build  cubic  model using function values and derivatives at trial
    points and we compare its prediction with actual value in  the  middle
    point

  -- ALGLIB --
     Copyright 15.06.2014 by Bochkanov Sergey
*************************************************************************/
void minnlcoptguardgradient(const minnlcstate &state, const double teststep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This  function  activates/deactivates nonsmoothness monitoring  option  of
the  OptGuard  integrity  checker. Smoothness  monitor  silently  observes
solution process and tries to detect ill-posed problems, i.e. ones with:
a) discontinuous target function (non-C0) and/or constraints
b) nonsmooth     target function (non-C1) and/or constraints

Smoothness monitoring does NOT interrupt optimization  even if it suspects
that your problem is nonsmooth. It just sets corresponding  flags  in  the
OptGuard report which can be retrieved after optimization is over.

Smoothness monitoring is a moderate overhead option which often adds  less
than 1% to the optimizer running time. Thus, you can use it even for large
scale problems.

NOTE: OptGuard does  NOT  guarantee  that  it  will  always  detect  C0/C1
      continuity violations.

      First, minor errors are hard to  catch - say, a 0.0001 difference in
      the model values at two sides of the gap may be due to discontinuity
      of the model - or simply because the model has changed.

      Second, C1-violations  are  especially  difficult  to  detect  in  a
      noninvasive way. The optimizer usually  performs  very  short  steps
      near the nonsmoothness, and differentiation  usually   introduces  a
      lot of numerical noise.  It  is  hard  to  tell  whether  some  tiny
      discontinuity in the slope is due to real nonsmoothness or just  due
      to numerical noise alone.

      Our top priority was to avoid false positives, so in some rare cases
      minor errors may went unnoticed (however, in most cases they can  be
      spotted with restart from different initial point).

INPUT PARAMETERS:
    state   -   algorithm state
    level   -   monitoring level:
                * 0 - monitoring is disabled
                * 1 - noninvasive low-overhead monitoring; function values
                      and/or gradients are recorded, but OptGuard does not
                      try to perform additional evaluations  in  order  to
                      get more information about suspicious locations.

=== EXPLANATION ==========================================================

One major source of headache during optimization  is  the  possibility  of
the coding errors in the target function/constraints (or their gradients).
Such  errors   most   often   manifest   themselves  as  discontinuity  or
nonsmoothness of the target/constraints.

Another frequent situation is when you try to optimize something involving
lots of min() and max() operations, i.e. nonsmooth target. Although not  a
coding error, it is nonsmoothness anyway - and smooth  optimizers  usually
stop right after encountering nonsmoothness, well before reaching solution.

OptGuard integrity checker helps you to catch such situations: it monitors
function values/gradients being passed  to  the  optimizer  and  tries  to
errors. Upon discovering suspicious pair of points it  raises  appropriate
flag (and allows you to continue optimization). When optimization is done,
you can study OptGuard result.

  -- ALGLIB --
     Copyright 21.11.2018 by Bochkanov Sergey
*************************************************************************/
void minnlcoptguardsmoothness(const minnlcstate &state, const ae_int_t level, const xparams _xparams = alglib::xdefault);
void minnlcoptguardsmoothness(const minnlcstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Results of OptGuard integrity check, should be called  after  optimization
session is over.

=== PRIMARY REPORT =======================================================

OptGuard performs several checks which are intended to catch common errors
in the implementation of nonlinear function/gradient:
* incorrect analytic gradient
* discontinuous (non-C0) target functions (constraints)
* nonsmooth     (non-C1) target functions (constraints)

Each of these checks is activated with appropriate function:
* minnlcoptguardgradient() for gradient verification
* minnlcoptguardsmoothness() for C0/C1 checks

Following flags are set when these errors are suspected:
* rep.badgradsuspected, and additionally:
  * rep.badgradfidx for specific function (Jacobian row) suspected
  * rep.badgradvidx for specific variable (Jacobian column) suspected
  * rep.badgradxbase, a point where gradient/Jacobian is tested
  * rep.badgraduser, user-provided gradient/Jacobian
  * rep.badgradnum, reference gradient/Jacobian obtained via numerical
    differentiation
* rep.nonc0suspected, and additionally:
  * rep.nonc0fidx - an index of specific function violating C0 continuity
* rep.nonc1suspected, and additionally
  * rep.nonc1fidx - an index of specific function violating C1 continuity
Here function index 0 means  target function, index 1  or  higher  denotes
nonlinear constraints.

=== ADDITIONAL REPORTS/LOGS ==============================================

Several different tests are performed to catch C0/C1 errors, you can  find
out specific test signaled error by looking to:
* rep.nonc0test0positive, for non-C0 test #0
* rep.nonc1test0positive, for non-C1 test #0
* rep.nonc1test1positive, for non-C1 test #1

Additional information (including line search logs)  can  be  obtained  by
means of:
* minnlcoptguardnonc1test0results()
* minnlcoptguardnonc1test1results()
which return detailed error reports, specific points where discontinuities
were found, and so on.

==========================================================================

INPUT PARAMETERS:
    state   -   algorithm state

OUTPUT PARAMETERS:
    rep     -   generic OptGuard report;  more  detailed  reports  can  be
                retrieved with other functions.

NOTE: false negatives (nonsmooth problems are not identified as  nonsmooth
      ones) are possible although unlikely.

      The reason  is  that  you  need  to  make several evaluations around
      nonsmoothness  in  order  to  accumulate  enough  information  about
      function curvature. Say, if you start right from the nonsmooth point,
      optimizer simply won't get enough data to understand what  is  going
      wrong before it terminates due to abrupt changes in the  derivative.
      It is also  possible  that  "unlucky"  step  will  move  us  to  the
      termination too quickly.

      Our current approach is to have less than 0.1%  false  negatives  in
      our test examples  (measured  with  multiple  restarts  from  random
      points), and to have exactly 0% false positives.

  -- ALGLIB --
     Copyright 21.11.2018 by Bochkanov Sergey
*************************************************************************/
void minnlcoptguardresults(const minnlcstate &state, optguardreport &rep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Detailed results of the OptGuard integrity check for nonsmoothness test #0

Nonsmoothness (non-C1) test #0 studies  function  values  (not  gradient!)
obtained during line searches and monitors  behavior  of  the  directional
derivative estimate.

This test is less powerful than test #1, but it does  not  depend  on  the
gradient values and thus it is more robust against artifacts introduced by
numerical differentiation.

Two reports are returned:
* a "strongest" one, corresponding  to  line   search  which  had  highest
  value of the nonsmoothness indicator
* a "longest" one, corresponding to line search which  had  more  function
  evaluations, and thus is more detailed

In both cases following fields are returned:

* positive - is TRUE  when test flagged suspicious point;  FALSE  if  test
  did not notice anything (in the latter cases fields below are empty).
* fidx - is an index of the function (0 for  target  function, 1 or higher
  for nonlinear constraints) which is suspected of being "non-C1"
* x0[], d[] - arrays of length N which store initial point  and  direction
  for line search (d[] can be normalized, but does not have to)
* stp[], f[] - arrays of length CNT which store step lengths and  function
  values at these points; f[i] is evaluated in x0+stp[i]*d.
* stpidxa, stpidxb - we  suspect  that  function  violates  C1  continuity
  between steps #stpidxa and #stpidxb (usually we have  stpidxb=stpidxa+3,
  with  most  likely  position  of  the  violation  between  stpidxa+1 and
  stpidxa+2.

==========================================================================
= SHORTLY SPEAKING: build a 2D plot of (stp,f) and look at it -  you  will
=                   see where C1 continuity is violated.
==========================================================================

INPUT PARAMETERS:
    state   -   algorithm state

OUTPUT PARAMETERS:
    strrep  -   C1 test #0 "strong" report
    lngrep  -   C1 test #0 "long" report

  -- ALGLIB --
     Copyright 21.11.2018 by Bochkanov Sergey
*************************************************************************/
void minnlcoptguardnonc1test0results(const minnlcstate &state, optguardnonc1test0report &strrep, optguardnonc1test0report &lngrep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Detailed results of the OptGuard integrity check for nonsmoothness test #1

Nonsmoothness (non-C1)  test  #1  studies  individual  components  of  the
gradient computed during line search.

When precise analytic gradient is provided this test is more powerful than
test #0  which  works  with  function  values  and  ignores  user-provided
gradient.  However,  test  #0  becomes  more   powerful   when   numerical
differentiation is employed (in such cases test #1 detects  higher  levels
of numerical noise and becomes too conservative).

This test also tells specific components of the gradient which violate  C1
continuity, which makes it more informative than #0, which just tells that
continuity is violated.

Two reports are returned:
* a "strongest" one, corresponding  to  line   search  which  had  highest
  value of the nonsmoothness indicator
* a "longest" one, corresponding to line search which  had  more  function
  evaluations, and thus is more detailed

In both cases following fields are returned:

* positive - is TRUE  when test flagged suspicious point;  FALSE  if  test
  did not notice anything (in the latter cases fields below are empty).
* fidx - is an index of the function (0 for  target  function, 1 or higher
  for nonlinear constraints) which is suspected of being "non-C1"
* vidx - is an index of the variable in [0,N) with nonsmooth derivative
* x0[], d[] - arrays of length N which store initial point  and  direction
  for line search (d[] can be normalized, but does not have to)
* stp[], g[] - arrays of length CNT which store step lengths and  gradient
  values at these points; g[i] is evaluated in  x0+stp[i]*d  and  contains
  vidx-th component of the gradient.
* stpidxa, stpidxb - we  suspect  that  function  violates  C1  continuity
  between steps #stpidxa and #stpidxb (usually we have  stpidxb=stpidxa+3,
  with  most  likely  position  of  the  violation  between  stpidxa+1 and
  stpidxa+2.

==========================================================================
= SHORTLY SPEAKING: build a 2D plot of (stp,f) and look at it -  you  will
=                   see where C1 continuity is violated.
==========================================================================

INPUT PARAMETERS:
    state   -   algorithm state

OUTPUT PARAMETERS:
    strrep  -   C1 test #1 "strong" report
    lngrep  -   C1 test #1 "long" report

  -- ALGLIB --
     Copyright 21.11.2018 by Bochkanov Sergey
*************************************************************************/
void minnlcoptguardnonc1test1results(const minnlcstate &state, optguardnonc1test1report &strrep, optguardnonc1test1report &lngrep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
MinNLC results:  the  solution  found,  completion  codes  and  additional
information.

If you activated OptGuard integrity checking functionality and want to get
OptGuard report, it can be retrieved with:
* minnlcoptguardresults() - for a primary report about (a) suspected C0/C1
  continuity violations and (b) errors in the analytic gradient.
* minnlcoptguardnonc1test0results() - for C1 continuity violation test #0,
  detailed line search log
* minnlcoptguardnonc1test1results() - for C1 continuity violation test #1,
  detailed line search log

INPUT PARAMETERS:
    State   -   algorithm state

OUTPUT PARAMETERS:
    X       -   array[0..N-1], solution
    Rep     -   optimization report, contains information about completion
                code, constraint violation at the solution and so on.

                You   should   check   rep.terminationtype  in  order   to
                distinguish successful termination from unsuccessful one:

                === FAILURE CODES ===
                * -8    internal  integrity control  detected  infinite or
                        NAN   values    in   function/gradient.   Abnormal
                        termination signalled.
                * -3    box  constraints are infeasible.
                        Note: infeasibility of  non-box  constraints  does
                              NOT trigger emergency completion;  you  have
                              to examine rep.bcerr/rep.lcerr/rep.nlcerr to
                              detect possibly inconsistent constraints.

                === SUCCESS CODES ===
                *  2   scaled step is no more than EpsX.
                *  5   MaxIts steps were taken.
                *  8   user   requested    algorithm    termination    via
                       minnlcrequesttermination(), last accepted point  is
                       returned.

                More information about fields of this  structure  can  be
                found in the comments on minnlcreport datatype.

  -- ALGLIB --
     Copyright 06.06.2014 by Bochkanov Sergey
*************************************************************************/
void minnlcresults(const minnlcstate &state, real_1d_array &x, minnlcreport &rep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
NLC results

Buffered implementation of MinNLCResults() which uses pre-allocated buffer
to store X[]. If buffer size is  too  small,  it  resizes  buffer.  It  is
intended to be used in the inner cycles of performance critical algorithms
where array reallocation penalty is too large to be ignored.

  -- ALGLIB --
     Copyright 28.11.2010 by Bochkanov Sergey
*************************************************************************/
void minnlcresultsbuf(const minnlcstate &state, real_1d_array &x, minnlcreport &rep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This subroutine submits request for termination of running  optimizer.  It
should be called from user-supplied callback when user decides that it  is
time to "smoothly" terminate optimization process.  As  result,  optimizer
stops at point which was "current accepted" when termination  request  was
submitted and returns error code 8 (successful termination).

INPUT PARAMETERS:
    State   -   optimizer structure

NOTE: after  request  for  termination  optimizer  may   perform   several
      additional calls to user-supplied callbacks. It does  NOT  guarantee
      to stop immediately - it just guarantees that these additional calls
      will be discarded later.

NOTE: calling this function on optimizer which is NOT running will have no
      effect.

NOTE: multiple calls to this function are possible. First call is counted,
      subsequent calls are silently ignored.

  -- ALGLIB --
     Copyright 08.10.2014 by Bochkanov Sergey
*************************************************************************/
void minnlcrequesttermination(const minnlcstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This subroutine restarts algorithm from new point.
All optimization parameters (including constraints) are left unchanged.

This  function  allows  to  solve multiple  optimization  problems  (which
must have  same number of dimensions) without object reallocation penalty.

INPUT PARAMETERS:
    State   -   structure previously allocated with MinNLCCreate call.
    X       -   new starting point.

  -- ALGLIB --
     Copyright 28.11.2010 by Bochkanov Sergey
*************************************************************************/
void minnlcrestartfrom(const minnlcstate &state, const real_1d_array &x, const xparams _xparams = alglib::xdefault);
#endif

#if defined(AE_COMPILE_MINBC) || !defined(AE_PARTIAL_BUILD)
/*************************************************************************
                     BOX CONSTRAINED OPTIMIZATION
          WITH FAST ACTIVATION OF MULTIPLE BOX CONSTRAINTS

DESCRIPTION:
The  subroutine  minimizes  function   F(x) of N arguments subject  to box
constraints (with some of box constraints actually being equality ones).

This optimizer uses algorithm similar to that of MinBLEIC (optimizer  with
general linear constraints), but presence of box-only  constraints  allows
us to use faster constraint activation strategies. On large-scale problems,
with multiple constraints active at the solution, this  optimizer  can  be
several times faster than BLEIC.

REQUIREMENTS:
* user must provide function value and gradient
* starting point X0 must be feasible or
  not too far away from the feasible set
* grad(f) must be Lipschitz continuous on a level set:
  L = { x : f(x)<=f(x0) }
* function must be defined everywhere on the feasible set F

USAGE:

Constrained optimization if far more complex than the unconstrained one.
Here we give very brief outline of the BC optimizer. We strongly recommend
you to read examples in the ALGLIB Reference Manual and to read ALGLIB User Guide
on optimization, which is available at http://www.alglib.net/optimization/

1. User initializes algorithm state with MinBCCreate() call

2. USer adds box constraints by calling MinBCSetBC() function.

3. User sets stopping conditions with MinBCSetCond().

4. User calls MinBCOptimize() function which takes algorithm  state and
   pointer (delegate, etc.) to callback function which calculates F/G.

5. User calls MinBCResults() to get solution

6. Optionally user may call MinBCRestartFrom() to solve another problem
   with same N but another starting point.
   MinBCRestartFrom() allows to reuse already initialized structure.


INPUT PARAMETERS:
    N       -   problem dimension, N>0:
                * if given, only leading N elements of X are used
                * if not given, automatically determined from size ofX
    X       -   starting point, array[N]:
                * it is better to set X to a feasible point
                * but X can be infeasible, in which case algorithm will try
                  to find feasible point first, using X as initial
                  approximation.

OUTPUT PARAMETERS:
    State   -   structure stores algorithm state

  -- ALGLIB --
     Copyright 28.11.2010 by Bochkanov Sergey
*************************************************************************/
void minbccreate(const ae_int_t n, const real_1d_array &x, minbcstate &state, const xparams _xparams = alglib::xdefault);
void minbccreate(const real_1d_array &x, minbcstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
The subroutine is finite difference variant of MinBCCreate().  It  uses
finite differences in order to differentiate target function.

Description below contains information which is specific to  this function
only. We recommend to read comments on MinBCCreate() in  order  to  get
more information about creation of BC optimizer.

INPUT PARAMETERS:
    N       -   problem dimension, N>0:
                * if given, only leading N elements of X are used
                * if not given, automatically determined from size of X
    X       -   starting point, array[0..N-1].
    DiffStep-   differentiation step, >0

OUTPUT PARAMETERS:
    State   -   structure which stores algorithm state

NOTES:
1. algorithm uses 4-point central formula for differentiation.
2. differentiation step along I-th axis is equal to DiffStep*S[I] where
   S[] is scaling vector which can be set by MinBCSetScale() call.
3. we recommend you to use moderate values of  differentiation  step.  Too
   large step will result in too large truncation  errors, while too small
   step will result in too large numerical  errors.  1.0E-6  can  be  good
   value to start with.
4. Numerical  differentiation  is   very   inefficient  -   one   gradient
   calculation needs 4*N function evaluations. This function will work for
   any N - either small (1...10), moderate (10...100) or  large  (100...).
   However, performance penalty will be too severe for any N's except  for
   small ones.
   We should also say that code which relies on numerical  differentiation
   is  less  robust and precise. CG needs exact gradient values. Imprecise
   gradient may slow  down  convergence, especially  on  highly  nonlinear
   problems.
   Thus  we  recommend to use this function for fast prototyping on small-
   dimensional problems only, and to implement analytical gradient as soon
   as possible.

  -- ALGLIB --
     Copyright 16.05.2011 by Bochkanov Sergey
*************************************************************************/
void minbccreatef(const ae_int_t n, const real_1d_array &x, const double diffstep, minbcstate &state, const xparams _xparams = alglib::xdefault);
void minbccreatef(const real_1d_array &x, const double diffstep, minbcstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets boundary constraints for BC optimizer.

Boundary constraints are inactive by default (after initial creation).
They are preserved after algorithm restart with MinBCRestartFrom().

INPUT PARAMETERS:
    State   -   structure stores algorithm state
    BndL    -   lower bounds, array[N].
                If some (all) variables are unbounded, you may specify
                very small number or -INF.
    BndU    -   upper bounds, array[N].
                If some (all) variables are unbounded, you may specify
                very large number or +INF.

NOTE 1: it is possible to specify BndL[i]=BndU[i]. In this case I-th
variable will be "frozen" at X[i]=BndL[i]=BndU[i].

NOTE 2: this solver has following useful properties:
* bound constraints are always satisfied exactly
* function is evaluated only INSIDE area specified by  bound  constraints,
  even  when  numerical  differentiation is used (algorithm adjusts  nodes
  according to boundary constraints)

  -- ALGLIB --
     Copyright 28.11.2010 by Bochkanov Sergey
*************************************************************************/
void minbcsetbc(const minbcstate &state, const real_1d_array &bndl, const real_1d_array &bndu, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets stopping conditions for the optimizer.

INPUT PARAMETERS:
    State   -   structure which stores algorithm state
    EpsG    -   >=0
                The  subroutine  finishes  its  work   if   the  condition
                |v|<EpsG is satisfied, where:
                * |.| means Euclidian norm
                * v - scaled gradient vector, v[i]=g[i]*s[i]
                * g - gradient
                * s - scaling coefficients set by MinBCSetScale()
    EpsF    -   >=0
                The  subroutine  finishes  its work if on k+1-th iteration
                the  condition  |F(k+1)-F(k)|<=EpsF*max{|F(k)|,|F(k+1)|,1}
                is satisfied.
    EpsX    -   >=0
                The subroutine finishes its work if  on  k+1-th  iteration
                the condition |v|<=EpsX is fulfilled, where:
                * |.| means Euclidian norm
                * v - scaled step vector, v[i]=dx[i]/s[i]
                * dx - step vector, dx=X(k+1)-X(k)
                * s - scaling coefficients set by MinBCSetScale()
    MaxIts  -   maximum number of iterations. If MaxIts=0, the  number  of
                iterations is unlimited.

Passing EpsG=0, EpsF=0 and EpsX=0 and MaxIts=0 (simultaneously) will lead
to automatic stopping criterion selection.

NOTE: when SetCond() called with non-zero MaxIts, BC solver may perform
      slightly more than MaxIts iterations. I.e., MaxIts  sets  non-strict
      limit on iterations count.

  -- ALGLIB --
     Copyright 28.11.2010 by Bochkanov Sergey
*************************************************************************/
void minbcsetcond(const minbcstate &state, const double epsg, const double epsf, const double epsx, const ae_int_t maxits, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets scaling coefficients for BC optimizer.

ALGLIB optimizers use scaling matrices to test stopping  conditions  (step
size and gradient are scaled before comparison with tolerances).  Scale of
the I-th variable is a translation invariant measure of:
a) "how large" the variable is
b) how large the step should be to make significant changes in the function

Scaling is also used by finite difference variant of the optimizer  - step
along I-th axis is equal to DiffStep*S[I].

In  most  optimizers  (and  in  the  BC  too)  scaling is NOT a form of
preconditioning. It just  affects  stopping  conditions.  You  should  set
preconditioner  by  separate  call  to  one  of  the  MinBCSetPrec...()
functions.

There is a special  preconditioning  mode, however,  which  uses   scaling
coefficients to form diagonal preconditioning matrix. You  can  turn  this
mode on, if you want.   But  you should understand that scaling is not the
same thing as preconditioning - these are two different, although  related
forms of tuning solver.

INPUT PARAMETERS:
    State   -   structure stores algorithm state
    S       -   array[N], non-zero scaling coefficients
                S[i] may be negative, sign doesn't matter.

  -- ALGLIB --
     Copyright 14.01.2011 by Bochkanov Sergey
*************************************************************************/
void minbcsetscale(const minbcstate &state, const real_1d_array &s, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Modification of the preconditioner: preconditioning is turned off.

INPUT PARAMETERS:
    State   -   structure which stores algorithm state

  -- ALGLIB --
     Copyright 13.10.2010 by Bochkanov Sergey
*************************************************************************/
void minbcsetprecdefault(const minbcstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Modification  of  the  preconditioner:  diagonal of approximate Hessian is
used.

INPUT PARAMETERS:
    State   -   structure which stores algorithm state
    D       -   diagonal of the approximate Hessian, array[0..N-1],
                (if larger, only leading N elements are used).

NOTE 1: D[i] should be positive. Exception will be thrown otherwise.

NOTE 2: you should pass diagonal of approximate Hessian - NOT ITS INVERSE.

  -- ALGLIB --
     Copyright 13.10.2010 by Bochkanov Sergey
*************************************************************************/
void minbcsetprecdiag(const minbcstate &state, const real_1d_array &d, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Modification of the preconditioner: scale-based diagonal preconditioning.

This preconditioning mode can be useful when you  don't  have  approximate
diagonal of Hessian, but you know that your  variables  are  badly  scaled
(for  example,  one  variable is in [1,10], and another in [1000,100000]),
and most part of the ill-conditioning comes from different scales of vars.

In this case simple  scale-based  preconditioner,  with H[i] = 1/(s[i]^2),
can greatly improve convergence.

IMPRTANT: you should set scale of your variables  with  MinBCSetScale()
call  (before  or after MinBCSetPrecScale() call). Without knowledge of
the scale of your variables scale-based preconditioner will be  just  unit
matrix.

INPUT PARAMETERS:
    State   -   structure which stores algorithm state

  -- ALGLIB --
     Copyright 13.10.2010 by Bochkanov Sergey
*************************************************************************/
void minbcsetprecscale(const minbcstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function turns on/off reporting.

INPUT PARAMETERS:
    State   -   structure which stores algorithm state
    NeedXRep-   whether iteration reports are needed or not

If NeedXRep is True, algorithm will call rep() callback function if  it is
provided to MinBCOptimize().

  -- ALGLIB --
     Copyright 28.11.2010 by Bochkanov Sergey
*************************************************************************/
void minbcsetxrep(const minbcstate &state, const bool needxrep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets maximum step length

INPUT PARAMETERS:
    State   -   structure which stores algorithm state
    StpMax  -   maximum step length, >=0. Set StpMax to 0.0,  if you don't
                want to limit step length.

Use this subroutine when you optimize target function which contains exp()
or  other  fast  growing  functions,  and optimization algorithm makes too
large  steps  which  lead   to overflow. This function allows us to reject
steps  that  are  too  large  (and  therefore  expose  us  to the possible
overflow) without actually calculating function value at the x+stp*d.

  -- ALGLIB --
     Copyright 02.04.2010 by Bochkanov Sergey
*************************************************************************/
void minbcsetstpmax(const minbcstate &state, const double stpmax, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function provides reverse communication interface
Reverse communication interface is not documented or recommended to use.
See below for functions which provide better documented API
*************************************************************************/
bool minbciteration(const minbcstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This family of functions is used to launcn iterations of nonlinear optimizer

These functions accept following parameters:
    state   -   algorithm state
    func    -   callback which calculates function (or merit function)
                value func at given point x
    grad    -   callback which calculates function (or merit function)
                value func and gradient grad at given point x
    rep     -   optional callback which is called after each iteration
                can be NULL
    ptr     -   optional pointer which is passed to func/grad/hess/jac/rep
                can be NULL

NOTES:

1. This function has two different implementations: one which  uses  exact
   (analytical) user-supplied gradient,  and one which uses function value
   only  and  numerically  differentiates  function  in  order  to  obtain
   gradient.

   Depending  on  the  specific  function  used to create optimizer object
   (either  MinBCCreate() for analytical gradient or  MinBCCreateF()
   for numerical differentiation) you should choose appropriate variant of
   MinBCOptimize() - one  which  accepts  function  AND gradient or one
   which accepts function ONLY.

   Be careful to choose variant of MinBCOptimize() which corresponds to
   your optimization scheme! Table below lists different  combinations  of
   callback (function/gradient) passed to MinBCOptimize()  and specific
   function used to create optimizer.


                     |         USER PASSED TO MinBCOptimize()
   CREATED WITH      |  function only   |  function and gradient
   ------------------------------------------------------------
   MinBCCreateF()    |     works               FAILS
   MinBCCreate()     |     FAILS               works

   Here "FAIL" denotes inappropriate combinations  of  optimizer  creation
   function  and  MinBCOptimize()  version.   Attemps   to   use   such
   combination (for  example,  to  create optimizer with MinBCCreateF()
   and  to  pass  gradient  information  to  MinCGOptimize()) will lead to
   exception being thrown. Either  you  did  not pass gradient when it WAS
   needed or you passed gradient when it was NOT needed.

  -- ALGLIB --
     Copyright 28.11.2010 by Bochkanov Sergey

*************************************************************************/
void minbcoptimize(minbcstate &state,
    void (*func)(const real_1d_array &x, double &func, void *ptr),
    void  (*rep)(const real_1d_array &x, double func, void *ptr) = NULL,
    void *ptr = NULL,
    const xparams _xparams = alglib::xdefault);
void minbcoptimize(minbcstate &state,
    void (*grad)(const real_1d_array &x, double &func, real_1d_array &grad, void *ptr),
    void  (*rep)(const real_1d_array &x, double func, void *ptr) = NULL,
    void *ptr = NULL,
    const xparams _xparams = alglib::xdefault);


/*************************************************************************
This  function  activates/deactivates verification  of  the  user-supplied
analytic gradient.

Upon  activation  of  this  option  OptGuard  integrity  checker  performs
numerical differentiation of your target function  at  the  initial  point
(note: future versions may also perform check  at  the  final  point)  and
compares numerical gradient with analytic one provided by you.

If difference is too large, an error flag is set and optimization  session
continues. After optimization session is over, you can retrieve the report
which  stores  both  gradients  and  specific  components  highlighted  as
suspicious by the OptGuard.

The primary OptGuard report can be retrieved with minbcoptguardresults().

IMPORTANT: gradient check is a high-overhead option which  will  cost  you
           about 3*N additional function evaluations. In many cases it may
           cost as much as the rest of the optimization session.

           YOU SHOULD NOT USE IT IN THE PRODUCTION CODE UNLESS YOU WANT TO
           CHECK DERIVATIVES PROVIDED BY SOME THIRD PARTY.

NOTE: unlike previous incarnation of the gradient checking code,  OptGuard
      does NOT interrupt optimization even if it discovers bad gradient.

INPUT PARAMETERS:
    State       -   structure used to store algorithm state
    TestStep    -   verification step used for numerical differentiation:
                    * TestStep=0 turns verification off
                    * TestStep>0 activates verification
                    You should carefully choose TestStep. Value  which  is
                    too large (so large that  function  behavior  is  non-
                    cubic at this scale) will lead  to  false  alarms. Too
                    short step will result in rounding  errors  dominating
                    numerical derivative.

                    You may use different step for different parameters by
                    means of setting scale with minbcsetscale().

=== EXPLANATION ==========================================================

In order to verify gradient algorithm performs following steps:
  * two trial steps are made to X[i]-TestStep*S[i] and X[i]+TestStep*S[i],
    where X[i] is i-th component of the initial point and S[i] is a  scale
    of i-th parameter
  * F(X) is evaluated at these trial points
  * we perform one more evaluation in the middle point of the interval
  * we  build  cubic  model using function values and derivatives at trial
    points and we compare its prediction with actual value in  the  middle
    point

  -- ALGLIB --
     Copyright 15.06.2014 by Bochkanov Sergey
*************************************************************************/
void minbcoptguardgradient(const minbcstate &state, const double teststep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This  function  activates/deactivates nonsmoothness monitoring  option  of
the  OptGuard  integrity  checker. Smoothness  monitor  silently  observes
solution process and tries to detect ill-posed problems, i.e. ones with:
a) discontinuous target function (non-C0)
b) nonsmooth     target function (non-C1)

Smoothness monitoring does NOT interrupt optimization  even if it suspects
that your problem is nonsmooth. It just sets corresponding  flags  in  the
OptGuard report which can be retrieved after optimization is over.

Smoothness monitoring is a moderate overhead option which often adds  less
than 1% to the optimizer running time. Thus, you can use it even for large
scale problems.

NOTE: OptGuard does  NOT  guarantee  that  it  will  always  detect  C0/C1
      continuity violations.

      First, minor errors are hard to  catch - say, a 0.0001 difference in
      the model values at two sides of the gap may be due to discontinuity
      of the model - or simply because the model has changed.

      Second, C1-violations  are  especially  difficult  to  detect  in  a
      noninvasive way. The optimizer usually  performs  very  short  steps
      near the nonsmoothness, and differentiation  usually   introduces  a
      lot of numerical noise.  It  is  hard  to  tell  whether  some  tiny
      discontinuity in the slope is due to real nonsmoothness or just  due
      to numerical noise alone.

      Our top priority was to avoid false positives, so in some rare cases
      minor errors may went unnoticed (however, in most cases they can  be
      spotted with restart from different initial point).

INPUT PARAMETERS:
    state   -   algorithm state
    level   -   monitoring level:
                * 0 - monitoring is disabled
                * 1 - noninvasive low-overhead monitoring; function values
                      and/or gradients are recorded, but OptGuard does not
                      try to perform additional evaluations  in  order  to
                      get more information about suspicious locations.

=== EXPLANATION ==========================================================

One major source of headache during optimization  is  the  possibility  of
the coding errors in the target function/constraints (or their gradients).
Such  errors   most   often   manifest   themselves  as  discontinuity  or
nonsmoothness of the target/constraints.

Another frequent situation is when you try to optimize something involving
lots of min() and max() operations, i.e. nonsmooth target. Although not  a
coding error, it is nonsmoothness anyway - and smooth  optimizers  usually
stop right after encountering nonsmoothness, well before reaching solution.

OptGuard integrity checker helps you to catch such situations: it monitors
function values/gradients being passed  to  the  optimizer  and  tries  to
errors. Upon discovering suspicious pair of points it  raises  appropriate
flag (and allows you to continue optimization). When optimization is done,
you can study OptGuard result.

  -- ALGLIB --
     Copyright 21.11.2018 by Bochkanov Sergey
*************************************************************************/
void minbcoptguardsmoothness(const minbcstate &state, const ae_int_t level, const xparams _xparams = alglib::xdefault);
void minbcoptguardsmoothness(const minbcstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Results of OptGuard integrity check, should be called  after  optimization
session is over.

=== PRIMARY REPORT =======================================================

OptGuard performs several checks which are intended to catch common errors
in the implementation of nonlinear function/gradient:
* incorrect analytic gradient
* discontinuous (non-C0) target functions (constraints)
* nonsmooth     (non-C1) target functions (constraints)

Each of these checks is activated with appropriate function:
* minbcoptguardgradient() for gradient verification
* minbcoptguardsmoothness() for C0/C1 checks

Following flags are set when these errors are suspected:
* rep.badgradsuspected, and additionally:
  * rep.badgradvidx for specific variable (gradient element) suspected
  * rep.badgradxbase, a point where gradient is tested
  * rep.badgraduser, user-provided gradient  (stored  as  2D  matrix  with
    single row in order to make  report  structure  compatible  with  more
    complex optimizers like MinNLC or MinLM)
  * rep.badgradnum,   reference    gradient    obtained    via   numerical
    differentiation (stored as  2D matrix with single row in order to make
    report structure compatible with more complex optimizers  like  MinNLC
    or MinLM)
* rep.nonc0suspected
* rep.nonc1suspected

=== ADDITIONAL REPORTS/LOGS ==============================================

Several different tests are performed to catch C0/C1 errors, you can  find
out specific test signaled error by looking to:
* rep.nonc0test0positive, for non-C0 test #0
* rep.nonc1test0positive, for non-C1 test #0
* rep.nonc1test1positive, for non-C1 test #1

Additional information (including line search logs)  can  be  obtained  by
means of:
* minbcoptguardnonc1test0results()
* minbcoptguardnonc1test1results()
which return detailed error reports, specific points where discontinuities
were found, and so on.

==========================================================================

INPUT PARAMETERS:
    state   -   algorithm state

OUTPUT PARAMETERS:
    rep     -   generic OptGuard report;  more  detailed  reports  can  be
                retrieved with other functions.

NOTE: false negatives (nonsmooth problems are not identified as  nonsmooth
      ones) are possible although unlikely.

      The reason  is  that  you  need  to  make several evaluations around
      nonsmoothness  in  order  to  accumulate  enough  information  about
      function curvature. Say, if you start right from the nonsmooth point,
      optimizer simply won't get enough data to understand what  is  going
      wrong before it terminates due to abrupt changes in the  derivative.
      It is also  possible  that  "unlucky"  step  will  move  us  to  the
      termination too quickly.

      Our current approach is to have less than 0.1%  false  negatives  in
      our test examples  (measured  with  multiple  restarts  from  random
      points), and to have exactly 0% false positives.

  -- ALGLIB --
     Copyright 21.11.2018 by Bochkanov Sergey
*************************************************************************/
void minbcoptguardresults(const minbcstate &state, optguardreport &rep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Detailed results of the OptGuard integrity check for nonsmoothness test #0

Nonsmoothness (non-C1) test #0 studies  function  values  (not  gradient!)
obtained during line searches and monitors  behavior  of  the  directional
derivative estimate.

This test is less powerful than test #1, but it does  not  depend  on  the
gradient values and thus it is more robust against artifacts introduced by
numerical differentiation.

Two reports are returned:
* a "strongest" one, corresponding  to  line   search  which  had  highest
  value of the nonsmoothness indicator
* a "longest" one, corresponding to line search which  had  more  function
  evaluations, and thus is more detailed

In both cases following fields are returned:

* positive - is TRUE  when test flagged suspicious point;  FALSE  if  test
  did not notice anything (in the latter cases fields below are empty).
* x0[], d[] - arrays of length N which store initial point  and  direction
  for line search (d[] can be normalized, but does not have to)
* stp[], f[] - arrays of length CNT which store step lengths and  function
  values at these points; f[i] is evaluated in x0+stp[i]*d.
* stpidxa, stpidxb - we  suspect  that  function  violates  C1  continuity
  between steps #stpidxa and #stpidxb (usually we have  stpidxb=stpidxa+3,
  with  most  likely  position  of  the  violation  between  stpidxa+1 and
  stpidxa+2.

==========================================================================
= SHORTLY SPEAKING: build a 2D plot of (stp,f) and look at it -  you  will
=                   see where C1 continuity is violated.
==========================================================================

INPUT PARAMETERS:
    state   -   algorithm state

OUTPUT PARAMETERS:
    strrep  -   C1 test #0 "strong" report
    lngrep  -   C1 test #0 "long" report

  -- ALGLIB --
     Copyright 21.11.2018 by Bochkanov Sergey
*************************************************************************/
void minbcoptguardnonc1test0results(const minbcstate &state, optguardnonc1test0report &strrep, optguardnonc1test0report &lngrep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Detailed results of the OptGuard integrity check for nonsmoothness test #1

Nonsmoothness (non-C1)  test  #1  studies  individual  components  of  the
gradient computed during line search.

When precise analytic gradient is provided this test is more powerful than
test #0  which  works  with  function  values  and  ignores  user-provided
gradient.  However,  test  #0  becomes  more   powerful   when   numerical
differentiation is employed (in such cases test #1 detects  higher  levels
of numerical noise and becomes too conservative).

This test also tells specific components of the gradient which violate  C1
continuity, which makes it more informative than #0, which just tells that
continuity is violated.

Two reports are returned:
* a "strongest" one, corresponding  to  line   search  which  had  highest
  value of the nonsmoothness indicator
* a "longest" one, corresponding to line search which  had  more  function
  evaluations, and thus is more detailed

In both cases following fields are returned:

* positive - is TRUE  when test flagged suspicious point;  FALSE  if  test
  did not notice anything (in the latter cases fields below are empty).
* vidx - is an index of the variable in [0,N) with nonsmooth derivative
* x0[], d[] - arrays of length N which store initial point  and  direction
  for line search (d[] can be normalized, but does not have to)
* stp[], g[] - arrays of length CNT which store step lengths and  gradient
  values at these points; g[i] is evaluated in  x0+stp[i]*d  and  contains
  vidx-th component of the gradient.
* stpidxa, stpidxb - we  suspect  that  function  violates  C1  continuity
  between steps #stpidxa and #stpidxb (usually we have  stpidxb=stpidxa+3,
  with  most  likely  position  of  the  violation  between  stpidxa+1 and
  stpidxa+2.

==========================================================================
= SHORTLY SPEAKING: build a 2D plot of (stp,f) and look at it -  you  will
=                   see where C1 continuity is violated.
==========================================================================

INPUT PARAMETERS:
    state   -   algorithm state

OUTPUT PARAMETERS:
    strrep  -   C1 test #1 "strong" report
    lngrep  -   C1 test #1 "long" report

  -- ALGLIB --
     Copyright 21.11.2018 by Bochkanov Sergey
*************************************************************************/
void minbcoptguardnonc1test1results(const minbcstate &state, optguardnonc1test1report &strrep, optguardnonc1test1report &lngrep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
BC results

INPUT PARAMETERS:
    State   -   algorithm state

OUTPUT PARAMETERS:
    X       -   array[0..N-1], solution
    Rep     -   optimization report. You should check Rep.TerminationType
                in  order  to  distinguish  successful  termination  from
                unsuccessful one:
                * -8    internal integrity control  detected  infinite or
                        NAN   values   in   function/gradient.   Abnormal
                        termination signalled.
                * -3   inconsistent constraints.
                *  1   relative function improvement is no more than EpsF.
                *  2   scaled step is no more than EpsX.
                *  4   scaled gradient norm is no more than EpsG.
                *  5   MaxIts steps was taken
                *  8   terminated by user who called minbcrequesttermination().
                       X contains point which was "current accepted"  when
                       termination request was submitted.
                More information about fields of this  structure  can  be
                found in the comments on MinBCReport datatype.

  -- ALGLIB --
     Copyright 28.11.2010 by Bochkanov Sergey
*************************************************************************/
void minbcresults(const minbcstate &state, real_1d_array &x, minbcreport &rep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
BC results

Buffered implementation of MinBCResults() which uses pre-allocated buffer
to store X[]. If buffer size is  too  small,  it  resizes  buffer.  It  is
intended to be used in the inner cycles of performance critical algorithms
where array reallocation penalty is too large to be ignored.

  -- ALGLIB --
     Copyright 28.11.2010 by Bochkanov Sergey
*************************************************************************/
void minbcresultsbuf(const minbcstate &state, real_1d_array &x, minbcreport &rep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This subroutine restarts algorithm from new point.
All optimization parameters (including constraints) are left unchanged.

This  function  allows  to  solve multiple  optimization  problems  (which
must have  same number of dimensions) without object reallocation penalty.

INPUT PARAMETERS:
    State   -   structure previously allocated with MinBCCreate call.
    X       -   new starting point.

  -- ALGLIB --
     Copyright 28.11.2010 by Bochkanov Sergey
*************************************************************************/
void minbcrestartfrom(const minbcstate &state, const real_1d_array &x, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This subroutine submits request for termination of running  optimizer.  It
should be called from user-supplied callback when user decides that it  is
time to "smoothly" terminate optimization process.  As  result,  optimizer
stops at point which was "current accepted" when termination  request  was
submitted and returns error code 8 (successful termination).

INPUT PARAMETERS:
    State   -   optimizer structure

NOTE: after  request  for  termination  optimizer  may   perform   several
      additional calls to user-supplied callbacks. It does  NOT  guarantee
      to stop immediately - it just guarantees that these additional calls
      will be discarded later.

NOTE: calling this function on optimizer which is NOT running will have no
      effect.

NOTE: multiple calls to this function are possible. First call is counted,
      subsequent calls are silently ignored.

  -- ALGLIB --
     Copyright 08.10.2014 by Bochkanov Sergey
*************************************************************************/
void minbcrequesttermination(const minbcstate &state, const xparams _xparams = alglib::xdefault);
#endif

#if defined(AE_COMPILE_MINNS) || !defined(AE_PARTIAL_BUILD)
/*************************************************************************
                  NONSMOOTH NONCONVEX OPTIMIZATION
            SUBJECT TO BOX/LINEAR/NONLINEAR-NONSMOOTH CONSTRAINTS

DESCRIPTION:

The  subroutine  minimizes  function   F(x)  of N arguments subject to any
combination of:
* bound constraints
* linear inequality constraints
* linear equality constraints
* nonlinear equality constraints Gi(x)=0
* nonlinear inequality constraints Hi(x)<=0

IMPORTANT: see MinNSSetAlgoAGS for important  information  on  performance
           restrictions of AGS solver.

REQUIREMENTS:
* starting point X0 must be feasible or not too far away from the feasible
  set
* F(), G(), H() are continuous, locally Lipschitz  and  continuously  (but
  not necessarily twice) differentiable in an open dense  subset  of  R^N.
  Functions F(), G() and H() may be nonsmooth and non-convex.
  Informally speaking, it means  that  functions  are  composed  of  large
  differentiable "patches" with nonsmoothness having  place  only  at  the
  boundaries between these "patches".
  Most real-life nonsmooth  functions  satisfy  these  requirements.  Say,
  anything which involves finite number of abs(), min() and max() is  very
  likely to pass the test.
  Say, it is possible to optimize anything of the following:
  * f=abs(x0)+2*abs(x1)
  * f=max(x0,x1)
  * f=sin(max(x0,x1)+abs(x2))
* for nonlinearly constrained problems: F()  must  be  bounded from  below
  without nonlinear constraints (this requirement is due to the fact that,
  contrary to box and linear constraints, nonlinear ones  require  special
  handling).
* user must provide function value and gradient for F(), H(), G()  at  all
  points where function/gradient can be calculated. If optimizer  requires
  value exactly at the boundary between "patches" (say, at x=0 for f=abs(x)),
  where gradient is not defined, user may resolve tie arbitrarily (in  our
  case - return +1 or -1 at its discretion).
* NS solver supports numerical differentiation, i.e. it may  differentiate
  your function for you,  but  it  results  in  2N  increase  of  function
  evaluations. Not recommended unless you solve really small problems. See
  minnscreatef() for more information on this functionality.

USAGE:

1. User initializes algorithm state with MinNSCreate() call  and   chooses
   what NLC solver to use. There is some solver which is used by  default,
   with default settings, but you should NOT rely on  default  choice.  It
   may change in future releases of ALGLIB without notice, and no one  can
   guarantee that new solver will be  able  to  solve  your  problem  with
   default settings.

   From the other side, if you choose solver explicitly, you can be pretty
   sure that it will work with new ALGLIB releases.

   In the current release following solvers can be used:
   * AGS solver (activated with MinNSSetAlgoAGS() function)

2. User adds boundary and/or linear and/or nonlinear constraints by  means
   of calling one of the following functions:
   a) MinNSSetBC() for boundary constraints
   b) MinNSSetLC() for linear constraints
   c) MinNSSetNLC() for nonlinear constraints
   You may combine (a), (b) and (c) in one optimization problem.

3. User sets scale of the variables with MinNSSetScale() function. It   is
   VERY important to set  scale  of  the  variables,  because  nonlinearly
   constrained problems are hard to solve when variables are badly scaled.

4. User sets stopping conditions with MinNSSetCond().

5. Finally, user calls MinNSOptimize()  function  which  takes   algorithm
   state and pointer (delegate, etc) to callback function which calculates
   F/G/H.

7. User calls MinNSResults() to get solution

8. Optionally user may call MinNSRestartFrom() to solve   another  problem
   with same N but another starting point. MinNSRestartFrom()  allows   to
   reuse already initialized structure.


INPUT PARAMETERS:
    N       -   problem dimension, N>0:
                * if given, only leading N elements of X are used
                * if not given, automatically determined from size of X
    X       -   starting point, array[N]:
                * it is better to set X to a feasible point
                * but X can be infeasible, in which case algorithm will try
                  to find feasible point first, using X as initial
                  approximation.

OUTPUT PARAMETERS:
    State   -   structure stores algorithm state

NOTE: minnscreatef() function may be used if  you  do  not  have  analytic
      gradient.   This   function  creates  solver  which  uses  numerical
      differentiation with user-specified step.

  -- ALGLIB --
     Copyright 18.05.2015 by Bochkanov Sergey
*************************************************************************/
void minnscreate(const ae_int_t n, const real_1d_array &x, minnsstate &state, const xparams _xparams = alglib::xdefault);
void minnscreate(const real_1d_array &x, minnsstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Version of minnscreatef() which uses numerical differentiation. I.e.,  you
do not have to calculate derivatives yourself. However, this version needs
2N times more function evaluations.

2-point differentiation formula is  used,  because  more  precise  4-point
formula is unstable when used on non-smooth functions.

INPUT PARAMETERS:
    N       -   problem dimension, N>0:
                * if given, only leading N elements of X are used
                * if not given, automatically determined from size of X
    X       -   starting point, array[N]:
                * it is better to set X to a feasible point
                * but X can be infeasible, in which case algorithm will try
                  to find feasible point first, using X as initial
                  approximation.
    DiffStep-   differentiation  step,  DiffStep>0.   Algorithm   performs
                numerical differentiation  with  step  for  I-th  variable
                being equal to DiffStep*S[I] (here S[] is a  scale vector,
                set by minnssetscale() function).
                Do not use  too  small  steps,  because  it  may  lead  to
                catastrophic cancellation during intermediate calculations.

OUTPUT PARAMETERS:
    State   -   structure stores algorithm state

  -- ALGLIB --
     Copyright 18.05.2015 by Bochkanov Sergey
*************************************************************************/
void minnscreatef(const ae_int_t n, const real_1d_array &x, const double diffstep, minnsstate &state, const xparams _xparams = alglib::xdefault);
void minnscreatef(const real_1d_array &x, const double diffstep, minnsstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets boundary constraints.

Boundary constraints are inactive by default (after initial creation).
They are preserved after algorithm restart with minnsrestartfrom().

INPUT PARAMETERS:
    State   -   structure stores algorithm state
    BndL    -   lower bounds, array[N].
                If some (all) variables are unbounded, you may specify
                very small number or -INF.
    BndU    -   upper bounds, array[N].
                If some (all) variables are unbounded, you may specify
                very large number or +INF.

NOTE 1: it is possible to specify BndL[i]=BndU[i]. In this case I-th
variable will be "frozen" at X[i]=BndL[i]=BndU[i].

NOTE 2: AGS solver has following useful properties:
* bound constraints are always satisfied exactly
* function is evaluated only INSIDE area specified by  bound  constraints,
  even  when  numerical  differentiation is used (algorithm adjusts  nodes
  according to boundary constraints)

  -- ALGLIB --
     Copyright 18.05.2015 by Bochkanov Sergey
*************************************************************************/
void minnssetbc(const minnsstate &state, const real_1d_array &bndl, const real_1d_array &bndu, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets linear constraints.

Linear constraints are inactive by default (after initial creation).
They are preserved after algorithm restart with minnsrestartfrom().

INPUT PARAMETERS:
    State   -   structure previously allocated with minnscreate() call.
    C       -   linear constraints, array[K,N+1].
                Each row of C represents one constraint, either equality
                or inequality (see below):
                * first N elements correspond to coefficients,
                * last element corresponds to the right part.
                All elements of C (including right part) must be finite.
    CT      -   type of constraints, array[K]:
                * if CT[i]>0, then I-th constraint is C[i,*]*x >= C[i,n+1]
                * if CT[i]=0, then I-th constraint is C[i,*]*x  = C[i,n+1]
                * if CT[i]<0, then I-th constraint is C[i,*]*x <= C[i,n+1]
    K       -   number of equality/inequality constraints, K>=0:
                * if given, only leading K elements of C/CT are used
                * if not given, automatically determined from sizes of C/CT

NOTE: linear (non-bound) constraints are satisfied only approximately:

* there always exists some minor violation (about current sampling  radius
  in magnitude during optimization, about EpsX in the solution) due to use
  of penalty method to handle constraints.
* numerical differentiation, if used, may  lead  to  function  evaluations
  outside  of the feasible  area,   because   algorithm  does  NOT  change
  numerical differentiation formula according to linear constraints.

If you want constraints to be  satisfied  exactly, try to reformulate your
problem  in  such  manner  that  all constraints will become boundary ones
(this kind of constraints is always satisfied exactly, both in  the  final
solution and in all intermediate points).

  -- ALGLIB --
     Copyright 18.05.2015 by Bochkanov Sergey
*************************************************************************/
void minnssetlc(const minnsstate &state, const real_2d_array &c, const integer_1d_array &ct, const ae_int_t k, const xparams _xparams = alglib::xdefault);
void minnssetlc(const minnsstate &state, const real_2d_array &c, const integer_1d_array &ct, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets nonlinear constraints.

In fact, this function sets NUMBER of nonlinear  constraints.  Constraints
itself (constraint functions) are passed to minnsoptimize() method.   This
method requires user-defined vector function F[]  and  its  Jacobian  J[],
where:
* first component of F[] and first row  of  Jacobian  J[]  correspond   to
  function being minimized
* next NLEC components of F[] (and rows  of  J)  correspond  to  nonlinear
  equality constraints G_i(x)=0
* next NLIC components of F[] (and rows  of  J)  correspond  to  nonlinear
  inequality constraints H_i(x)<=0

NOTE: you may combine nonlinear constraints with linear/boundary ones.  If
      your problem has mixed constraints, you  may explicitly specify some
      of them as linear ones. It may help optimizer to  handle  them  more
      efficiently.

INPUT PARAMETERS:
    State   -   structure previously allocated with minnscreate() call.
    NLEC    -   number of Non-Linear Equality Constraints (NLEC), >=0
    NLIC    -   number of Non-Linear Inquality Constraints (NLIC), >=0

NOTE 1: nonlinear constraints are satisfied only  approximately!   It   is
        possible   that  algorithm  will  evaluate  function  outside   of
        the feasible area!

NOTE 2: algorithm scales variables  according  to   scale   specified   by
        minnssetscale()  function,  so  it can handle problems with  badly
        scaled variables (as long as we KNOW their scales).

        However,  there  is  no  way  to  automatically  scale   nonlinear
        constraints Gi(x) and Hi(x). Inappropriate scaling  of  Gi/Hi  may
        ruin convergence. Solving problem with  constraint  "1000*G0(x)=0"
        is NOT same as solving it with constraint "0.001*G0(x)=0".

        It  means  that  YOU  are  the  one who is responsible for correct
        scaling of nonlinear constraints Gi(x) and Hi(x). We recommend you
        to scale nonlinear constraints in such way that I-th component  of
        dG/dX (or dH/dx) has approximately unit  magnitude  (for  problems
        with unit scale)  or  has  magnitude approximately equal to 1/S[i]
        (where S is a scale set by minnssetscale() function).

NOTE 3: nonlinear constraints are always hard to handle,  no  matter  what
        algorithm you try to use. Even basic box/linear constraints modify
        function  curvature   by  adding   valleys  and  ridges.  However,
        nonlinear constraints add valleys which are very  hard  to  follow
        due to their "curved" nature.

        It means that optimization with single nonlinear constraint may be
        significantly slower than optimization with multiple linear  ones.
        It is normal situation, and we recommend you to  carefully  choose
        Rho parameter of minnssetalgoags(), because too  large  value  may
        slow down convergence.


  -- ALGLIB --
     Copyright 18.05.2015 by Bochkanov Sergey
*************************************************************************/
void minnssetnlc(const minnsstate &state, const ae_int_t nlec, const ae_int_t nlic, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets stopping conditions for iterations of optimizer.

INPUT PARAMETERS:
    State   -   structure which stores algorithm state
    EpsX    -   >=0
                The AGS solver finishes its work if  on  k+1-th  iteration
                sampling radius decreases below EpsX.
    MaxIts  -   maximum number of iterations. If MaxIts=0, the  number  of
                iterations is unlimited.

Passing EpsX=0  and  MaxIts=0  (simultaneously)  will  lead  to  automatic
stopping criterion selection. We do not recommend you to rely  on  default
choice in production code.

  -- ALGLIB --
     Copyright 18.05.2015 by Bochkanov Sergey
*************************************************************************/
void minnssetcond(const minnsstate &state, const double epsx, const ae_int_t maxits, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets scaling coefficients for NLC optimizer.

ALGLIB optimizers use scaling matrices to test stopping  conditions  (step
size and gradient are scaled before comparison with tolerances).  Scale of
the I-th variable is a translation invariant measure of:
a) "how large" the variable is
b) how large the step should be to make significant changes in the function

Scaling is also used by finite difference variant of the optimizer  - step
along I-th axis is equal to DiffStep*S[I].

INPUT PARAMETERS:
    State   -   structure stores algorithm state
    S       -   array[N], non-zero scaling coefficients
                S[i] may be negative, sign doesn't matter.

  -- ALGLIB --
     Copyright 18.05.2015 by Bochkanov Sergey
*************************************************************************/
void minnssetscale(const minnsstate &state, const real_1d_array &s, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function tells MinNS unit to use  AGS  (adaptive  gradient  sampling)
algorithm for nonsmooth constrained  optimization.  This  algorithm  is  a
slight modification of one described in  "An  Adaptive  Gradient  Sampling
Algorithm for Nonsmooth Optimization" by Frank E. Curtisy and Xiaocun Quez.

This optimizer has following benefits and drawbacks:
+ robustness; it can be used with nonsmooth and nonconvex functions.
+ relatively easy tuning; most of the metaparameters are easy to select.
- it has convergence of steepest descent, slower than CG/LBFGS.
- each iteration involves evaluation of ~2N gradient values  and  solution
  of 2Nx2N quadratic programming problem, which  limits  applicability  of
  algorithm by small-scale problems (up to 50-100).

IMPORTANT: this  algorithm  has  convergence  guarantees,   i.e.  it  will
           steadily move towards some stationary point of the function.

           However, "stationary point" does not  always  mean  "solution".
           Nonsmooth problems often have "flat spots",  i.e.  areas  where
           function do not change at all. Such "flat spots" are stationary
           points by definition, and algorithm may be caught here.

           Nonsmooth CONVEX tasks are not prone to  this  problem. Say, if
           your function has form f()=MAX(f0,f1,...), and f_i are  convex,
           then f() is convex too and you have guaranteed  convergence  to
           solution.

INPUT PARAMETERS:
    State   -   structure which stores algorithm state
    Radius  -   initial sampling radius, >=0.

                Internally multiplied  by  vector of  per-variable  scales
                specified by minnssetscale()).

                You should select relatively large sampling radius, roughly
                proportional to scaled length of the first  steps  of  the
                algorithm. Something close to 0.1 in magnitude  should  be
                good for most problems.

                AGS solver can automatically decrease radius, so too large
                radius is  not a problem (assuming that you  won't  choose
                so large radius that algorithm  will  sample  function  in
                too far away points, where gradient value is irrelevant).

                Too small radius won't cause algorithm to fail, but it may
                slow down algorithm (it may  have  to  perform  too  short
                steps).
    Penalty -   penalty coefficient for nonlinear constraints:
                * for problem with nonlinear constraints  should  be  some
                  problem-specific  positive   value,  large  enough  that
                  penalty term changes shape of the function.
                  Starting  from  some  problem-specific   value   penalty
                  coefficient becomes  large  enough  to  exactly  enforce
                  nonlinear constraints;  larger  values  do  not  improve
                  precision.
                  Increasing it too much may slow down convergence, so you
                  should choose it carefully.
                * can be zero for problems WITHOUT  nonlinear  constraints
                  (i.e. for unconstrained ones or ones with  just  box  or
                  linear constraints)
                * if you specify zero value for problem with at least  one
                  nonlinear  constraint,  algorithm  will  terminate  with
                  error code -1.

ALGORITHM OUTLINE

The very basic outline of unconstrained AGS algorithm is given below:

0. If sampling radius is below EpsX  or  we  performed  more  then  MaxIts
   iterations - STOP.
1. sample O(N) gradient values at random locations  around  current point;
   informally speaking, this sample is an implicit piecewise  linear model
   of the function, although algorithm formulation does  not  mention that
   explicitly
2. solve quadratic programming problem in order to find descent direction
3. if QP solver tells us that we  are  near  solution,  decrease  sampling
   radius and move to (0)
4. perform backtracking line search
5. after moving to new point, goto (0)

As for the constraints:
* box constraints are handled exactly  by  modification  of  the  function
  being minimized
* linear/nonlinear constraints are handled by adding L1  penalty.  Because
  our solver can handle nonsmoothness, we can  use  L1  penalty  function,
  which is an exact one  (i.e.  exact  solution  is  returned  under  such
  penalty).
* penalty coefficient for  linear  constraints  is  chosen  automatically;
  however, penalty coefficient for nonlinear constraints must be specified
  by user.

  -- ALGLIB --
     Copyright 18.05.2015 by Bochkanov Sergey
*************************************************************************/
void minnssetalgoags(const minnsstate &state, const double radius, const double penalty, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function turns on/off reporting.

INPUT PARAMETERS:
    State   -   structure which stores algorithm state
    NeedXRep-   whether iteration reports are needed or not

If NeedXRep is True, algorithm will call rep() callback function if  it is
provided to minnsoptimize().

  -- ALGLIB --
     Copyright 28.11.2010 by Bochkanov Sergey
*************************************************************************/
void minnssetxrep(const minnsstate &state, const bool needxrep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This subroutine submits request for termination of running  optimizer.  It
should be called from user-supplied callback when user decides that it  is
time to "smoothly" terminate optimization process.  As  result,  optimizer
stops at point which was "current accepted" when termination  request  was
submitted and returns error code 8 (successful termination).

INPUT PARAMETERS:
    State   -   optimizer structure

NOTE: after  request  for  termination  optimizer  may   perform   several
      additional calls to user-supplied callbacks. It does  NOT  guarantee
      to stop immediately - it just guarantees that these additional calls
      will be discarded later.

NOTE: calling this function on optimizer which is NOT running will have no
      effect.

NOTE: multiple calls to this function are possible. First call is counted,
      subsequent calls are silently ignored.

  -- ALGLIB --
     Copyright 18.05.2015 by Bochkanov Sergey
*************************************************************************/
void minnsrequesttermination(const minnsstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function provides reverse communication interface
Reverse communication interface is not documented or recommended to use.
See below for functions which provide better documented API
*************************************************************************/
bool minnsiteration(const minnsstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This family of functions is used to launcn iterations of nonlinear optimizer

These functions accept following parameters:
    state   -   algorithm state
    fvec    -   callback which calculates function vector fi[]
                at given point x
    jac     -   callback which calculates function vector fi[]
                and Jacobian jac at given point x
    rep     -   optional callback which is called after each iteration
                can be NULL
    ptr     -   optional pointer which is passed to func/grad/hess/jac/rep
                can be NULL


NOTES:

1. This function has two different implementations: one which  uses  exact
   (analytical) user-supplied Jacobian, and one which uses  only  function
   vector and numerically  differentiates  function  in  order  to  obtain
   gradient.

   Depending  on  the  specific  function  used to create optimizer object
   you should choose appropriate variant of  minnsoptimize() -  one  which
   accepts function AND Jacobian or one which accepts ONLY function.

   Be careful to choose variant of minnsoptimize()  which  corresponds  to
   your optimization scheme! Table below lists different  combinations  of
   callback (function/gradient) passed to minnsoptimize()    and  specific
   function used to create optimizer.


                     |         USER PASSED TO minnsoptimize()
   CREATED WITH      |  function only   |  function and gradient
   ------------------------------------------------------------
   minnscreatef()    |     works               FAILS
   minnscreate()     |     FAILS               works

   Here "FAILS" denotes inappropriate combinations  of  optimizer creation
   function  and  minnsoptimize()  version.   Attemps   to    use     such
   combination will lead to exception. Either  you  did  not pass gradient
   when it WAS needed or you passed gradient when it was NOT needed.

  -- ALGLIB --
     Copyright 18.05.2015 by Bochkanov Sergey

*************************************************************************/
void minnsoptimize(minnsstate &state,
    void (*fvec)(const real_1d_array &x, real_1d_array &fi, void *ptr),
    void  (*rep)(const real_1d_array &x, double func, void *ptr) = NULL,
    void *ptr = NULL,
    const xparams _xparams = alglib::xdefault);
void minnsoptimize(minnsstate &state,
    void  (*jac)(const real_1d_array &x, real_1d_array &fi, real_2d_array &jac, void *ptr),
    void  (*rep)(const real_1d_array &x, double func, void *ptr) = NULL,
    void *ptr = NULL,
    const xparams _xparams = alglib::xdefault);


/*************************************************************************
MinNS results

INPUT PARAMETERS:
    State   -   algorithm state

OUTPUT PARAMETERS:
    X       -   array[0..N-1], solution
    Rep     -   optimization report. You should check Rep.TerminationType
                in  order  to  distinguish  successful  termination  from
                unsuccessful one:
                * -8   internal integrity control  detected  infinite  or
                       NAN   values   in   function/gradient.    Abnormal
                       termination signalled.
                * -3   box constraints are inconsistent
                * -1   inconsistent parameters were passed:
                       * penalty parameter for minnssetalgoags() is zero,
                         but we have nonlinear constraints set by minnssetnlc()
                *  2   sampling radius decreased below epsx
                *  7    stopping conditions are too stringent,
                        further improvement is impossible,
                        X contains best point found so far.
                *  8    User requested termination via minnsrequesttermination()

  -- ALGLIB --
     Copyright 18.05.2015 by Bochkanov Sergey
*************************************************************************/
void minnsresults(const minnsstate &state, real_1d_array &x, minnsreport &rep, const xparams _xparams = alglib::xdefault);


/*************************************************************************

Buffered implementation of minnsresults() which uses pre-allocated  buffer
to store X[]. If buffer size is  too  small,  it  resizes  buffer.  It  is
intended to be used in the inner cycles of performance critical algorithms
where array reallocation penalty is too large to be ignored.

  -- ALGLIB --
     Copyright 18.05.2015 by Bochkanov Sergey
*************************************************************************/
void minnsresultsbuf(const minnsstate &state, real_1d_array &x, minnsreport &rep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This subroutine restarts algorithm from new point.
All optimization parameters (including constraints) are left unchanged.

This  function  allows  to  solve multiple  optimization  problems  (which
must have  same number of dimensions) without object reallocation penalty.

INPUT PARAMETERS:
    State   -   structure previously allocated with minnscreate() call.
    X       -   new starting point.

  -- ALGLIB --
     Copyright 18.05.2015 by Bochkanov Sergey
*************************************************************************/
void minnsrestartfrom(const minnsstate &state, const real_1d_array &x, const xparams _xparams = alglib::xdefault);
#endif

#if defined(AE_COMPILE_MINCOMP) || !defined(AE_PARTIAL_BUILD)
/*************************************************************************
Obsolete function, use MinLBFGSSetPrecDefault() instead.

  -- ALGLIB --
     Copyright 13.10.2010 by Bochkanov Sergey
*************************************************************************/
void minlbfgssetdefaultpreconditioner(const minlbfgsstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Obsolete function, use MinLBFGSSetCholeskyPreconditioner() instead.

  -- ALGLIB --
     Copyright 13.10.2010 by Bochkanov Sergey
*************************************************************************/
void minlbfgssetcholeskypreconditioner(const minlbfgsstate &state, const real_2d_array &p, const bool isupper, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This is obsolete function which was used by previous version of the  BLEIC
optimizer. It does nothing in the current version of BLEIC.

  -- ALGLIB --
     Copyright 28.11.2010 by Bochkanov Sergey
*************************************************************************/
void minbleicsetbarrierwidth(const minbleicstate &state, const double mu, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This is obsolete function which was used by previous version of the  BLEIC
optimizer. It does nothing in the current version of BLEIC.

  -- ALGLIB --
     Copyright 28.11.2010 by Bochkanov Sergey
*************************************************************************/
void minbleicsetbarrierdecay(const minbleicstate &state, const double mudecay, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Obsolete optimization algorithm.
Was replaced by MinBLEIC subpackage.

  -- ALGLIB --
     Copyright 25.03.2010 by Bochkanov Sergey
*************************************************************************/
void minasacreate(const ae_int_t n, const real_1d_array &x, const real_1d_array &bndl, const real_1d_array &bndu, minasastate &state, const xparams _xparams = alglib::xdefault);
void minasacreate(const real_1d_array &x, const real_1d_array &bndl, const real_1d_array &bndu, minasastate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Obsolete optimization algorithm.
Was replaced by MinBLEIC subpackage.

  -- ALGLIB --
     Copyright 02.04.2010 by Bochkanov Sergey
*************************************************************************/
void minasasetcond(const minasastate &state, const double epsg, const double epsf, const double epsx, const ae_int_t maxits, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Obsolete optimization algorithm.
Was replaced by MinBLEIC subpackage.

  -- ALGLIB --
     Copyright 02.04.2010 by Bochkanov Sergey
*************************************************************************/
void minasasetxrep(const minasastate &state, const bool needxrep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Obsolete optimization algorithm.
Was replaced by MinBLEIC subpackage.

  -- ALGLIB --
     Copyright 02.04.2010 by Bochkanov Sergey
*************************************************************************/
void minasasetalgorithm(const minasastate &state, const ae_int_t algotype, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Obsolete optimization algorithm.
Was replaced by MinBLEIC subpackage.

  -- ALGLIB --
     Copyright 02.04.2010 by Bochkanov Sergey
*************************************************************************/
void minasasetstpmax(const minasastate &state, const double stpmax, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function provides reverse communication interface
Reverse communication interface is not documented or recommended to use.
See below for functions which provide better documented API
*************************************************************************/
bool minasaiteration(const minasastate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This family of functions is used to launcn iterations of nonlinear optimizer

These functions accept following parameters:
    state   -   algorithm state
    grad    -   callback which calculates function (or merit function)
                value func and gradient grad at given point x
    rep     -   optional callback which is called after each iteration
                can be NULL
    ptr     -   optional pointer which is passed to func/grad/hess/jac/rep
                can be NULL


  -- ALGLIB --
     Copyright 20.03.2009 by Bochkanov Sergey

*************************************************************************/
void minasaoptimize(minasastate &state,
    void (*grad)(const real_1d_array &x, double &func, real_1d_array &grad, void *ptr),
    void  (*rep)(const real_1d_array &x, double func, void *ptr) = NULL,
    void *ptr = NULL,
    const xparams _xparams = alglib::xdefault);


/*************************************************************************
Obsolete optimization algorithm.
Was replaced by MinBLEIC subpackage.

  -- ALGLIB --
     Copyright 20.03.2009 by Bochkanov Sergey
*************************************************************************/
void minasaresults(const minasastate &state, real_1d_array &x, minasareport &rep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Obsolete optimization algorithm.
Was replaced by MinBLEIC subpackage.

  -- ALGLIB --
     Copyright 20.03.2009 by Bochkanov Sergey
*************************************************************************/
void minasaresultsbuf(const minasastate &state, real_1d_array &x, minasareport &rep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Obsolete optimization algorithm.
Was replaced by MinBLEIC subpackage.

  -- ALGLIB --
     Copyright 30.07.2010 by Bochkanov Sergey
*************************************************************************/
void minasarestartfrom(const minasastate &state, const real_1d_array &x, const real_1d_array &bndl, const real_1d_array &bndu, const xparams _xparams = alglib::xdefault);
#endif

#if defined(AE_COMPILE_MINCG) || !defined(AE_PARTIAL_BUILD)
/*************************************************************************
        NONLINEAR CONJUGATE GRADIENT METHOD

DESCRIPTION:
The subroutine minimizes function F(x) of N arguments by using one of  the
nonlinear conjugate gradient methods.

These CG methods are globally convergent (even on non-convex functions) as
long as grad(f) is Lipschitz continuous in  a  some  neighborhood  of  the
L = { x : f(x)<=f(x0) }.


REQUIREMENTS:
Algorithm will request following information during its operation:
* function value F and its gradient G (simultaneously) at given point X


USAGE:
1. User initializes algorithm state with MinCGCreate() call
2. User tunes solver parameters with MinCGSetCond(), MinCGSetStpMax() and
   other functions
3. User calls MinCGOptimize() function which takes algorithm  state   and
   pointer (delegate, etc.) to callback function which calculates F/G.
4. User calls MinCGResults() to get solution
5. Optionally, user may call MinCGRestartFrom() to solve another  problem
   with same N but another starting point and/or another function.
   MinCGRestartFrom() allows to reuse already initialized structure.


INPUT PARAMETERS:
    N       -   problem dimension, N>0:
                * if given, only leading N elements of X are used
                * if not given, automatically determined from size of X
    X       -   starting point, array[0..N-1].

OUTPUT PARAMETERS:
    State   -   structure which stores algorithm state

  -- ALGLIB --
     Copyright 25.03.2010 by Bochkanov Sergey
*************************************************************************/
void mincgcreate(const ae_int_t n, const real_1d_array &x, mincgstate &state, const xparams _xparams = alglib::xdefault);
void mincgcreate(const real_1d_array &x, mincgstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
The subroutine is finite difference variant of MinCGCreate(). It uses
finite differences in order to differentiate target function.

Description below contains information which is specific to this function
only. We recommend to read comments on MinCGCreate() in order to get more
information about creation of CG optimizer.

INPUT PARAMETERS:
    N       -   problem dimension, N>0:
                * if given, only leading N elements of X are used
                * if not given, automatically determined from size of X
    X       -   starting point, array[0..N-1].
    DiffStep-   differentiation step, >0

OUTPUT PARAMETERS:
    State   -   structure which stores algorithm state

NOTES:
1. algorithm uses 4-point central formula for differentiation.
2. differentiation step along I-th axis is equal to DiffStep*S[I] where
   S[] is scaling vector which can be set by MinCGSetScale() call.
3. we recommend you to use moderate values of  differentiation  step.  Too
   large step will result in too large truncation  errors, while too small
   step will result in too large numerical  errors.  1.0E-6  can  be  good
   value to start with.
4. Numerical  differentiation  is   very   inefficient  -   one   gradient
   calculation needs 4*N function evaluations. This function will work for
   any N - either small (1...10), moderate (10...100) or  large  (100...).
   However, performance penalty will be too severe for any N's except  for
   small ones.
   We should also say that code which relies on numerical  differentiation
   is  less  robust  and  precise.  L-BFGS  needs  exact  gradient values.
   Imprecise  gradient may slow down  convergence,  especially  on  highly
   nonlinear problems.
   Thus  we  recommend to use this function for fast prototyping on small-
   dimensional problems only, and to implement analytical gradient as soon
   as possible.

  -- ALGLIB --
     Copyright 16.05.2011 by Bochkanov Sergey
*************************************************************************/
void mincgcreatef(const ae_int_t n, const real_1d_array &x, const double diffstep, mincgstate &state, const xparams _xparams = alglib::xdefault);
void mincgcreatef(const real_1d_array &x, const double diffstep, mincgstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets stopping conditions for CG optimization algorithm.

INPUT PARAMETERS:
    State   -   structure which stores algorithm state
    EpsG    -   >=0
                The  subroutine  finishes  its  work   if   the  condition
                |v|<EpsG is satisfied, where:
                * |.| means Euclidian norm
                * v - scaled gradient vector, v[i]=g[i]*s[i]
                * g - gradient
                * s - scaling coefficients set by MinCGSetScale()
    EpsF    -   >=0
                The  subroutine  finishes  its work if on k+1-th iteration
                the  condition  |F(k+1)-F(k)|<=EpsF*max{|F(k)|,|F(k+1)|,1}
                is satisfied.
    EpsX    -   >=0
                The subroutine finishes its work if  on  k+1-th  iteration
                the condition |v|<=EpsX is fulfilled, where:
                * |.| means Euclidian norm
                * v - scaled step vector, v[i]=dx[i]/s[i]
                * dx - ste pvector, dx=X(k+1)-X(k)
                * s - scaling coefficients set by MinCGSetScale()
    MaxIts  -   maximum number of iterations. If MaxIts=0, the  number  of
                iterations is unlimited.

Passing EpsG=0, EpsF=0, EpsX=0 and MaxIts=0 (simultaneously) will lead to
automatic stopping criterion selection (small EpsX).

  -- ALGLIB --
     Copyright 02.04.2010 by Bochkanov Sergey
*************************************************************************/
void mincgsetcond(const mincgstate &state, const double epsg, const double epsf, const double epsx, const ae_int_t maxits, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets scaling coefficients for CG optimizer.

ALGLIB optimizers use scaling matrices to test stopping  conditions  (step
size and gradient are scaled before comparison with tolerances).  Scale of
the I-th variable is a translation invariant measure of:
a) "how large" the variable is
b) how large the step should be to make significant changes in the function

Scaling is also used by finite difference variant of CG optimizer  -  step
along I-th axis is equal to DiffStep*S[I].

In   most   optimizers  (and  in  the  CG  too)  scaling is NOT a form  of
preconditioning. It just  affects  stopping  conditions.  You  should  set
preconditioner by separate call to one of the MinCGSetPrec...() functions.

There  is  special  preconditioning  mode, however,  which  uses   scaling
coefficients to form diagonal preconditioning matrix. You  can  turn  this
mode on, if you want.   But  you should understand that scaling is not the
same thing as preconditioning - these are two different, although  related
forms of tuning solver.

INPUT PARAMETERS:
    State   -   structure stores algorithm state
    S       -   array[N], non-zero scaling coefficients
                S[i] may be negative, sign doesn't matter.

  -- ALGLIB --
     Copyright 14.01.2011 by Bochkanov Sergey
*************************************************************************/
void mincgsetscale(const mincgstate &state, const real_1d_array &s, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function turns on/off reporting.

INPUT PARAMETERS:
    State   -   structure which stores algorithm state
    NeedXRep-   whether iteration reports are needed or not

If NeedXRep is True, algorithm will call rep() callback function if  it is
provided to MinCGOptimize().

  -- ALGLIB --
     Copyright 02.04.2010 by Bochkanov Sergey
*************************************************************************/
void mincgsetxrep(const mincgstate &state, const bool needxrep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets CG algorithm.

INPUT PARAMETERS:
    State   -   structure which stores algorithm state
    CGType  -   algorithm type:
                * -1    automatic selection of the best algorithm
                * 0     DY (Dai and Yuan) algorithm
                * 1     Hybrid DY-HS algorithm

  -- ALGLIB --
     Copyright 02.04.2010 by Bochkanov Sergey
*************************************************************************/
void mincgsetcgtype(const mincgstate &state, const ae_int_t cgtype, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets maximum step length

INPUT PARAMETERS:
    State   -   structure which stores algorithm state
    StpMax  -   maximum step length, >=0. Set StpMax to 0.0,  if you don't
                want to limit step length.

Use this subroutine when you optimize target function which contains exp()
or  other  fast  growing  functions,  and optimization algorithm makes too
large  steps  which  leads  to overflow. This function allows us to reject
steps  that  are  too  large  (and  therefore  expose  us  to the possible
overflow) without actually calculating function value at the x+stp*d.

  -- ALGLIB --
     Copyright 02.04.2010 by Bochkanov Sergey
*************************************************************************/
void mincgsetstpmax(const mincgstate &state, const double stpmax, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function allows to suggest initial step length to the CG algorithm.

Suggested  step  length  is used as starting point for the line search. It
can be useful when you have  badly  scaled  problem,  i.e.  when  ||grad||
(which is used as initial estimate for the first step) is many  orders  of
magnitude different from the desired step.

Line search  may  fail  on  such problems without good estimate of initial
step length. Imagine, for example, problem with ||grad||=10^50 and desired
step equal to 0.1 Line  search function will use 10^50  as  initial  step,
then  it  will  decrease step length by 2 (up to 20 attempts) and will get
10^44, which is still too large.

This function allows us to tell than line search should  be  started  from
some moderate step length, like 1.0, so algorithm will be able  to  detect
desired step length in a several searches.

Default behavior (when no step is suggested) is to use preconditioner,  if
it is available, to generate initial estimate of step length.

This function influences only first iteration of algorithm. It  should  be
called between MinCGCreate/MinCGRestartFrom() call and MinCGOptimize call.
Suggested step is ignored if you have preconditioner.

INPUT PARAMETERS:
    State   -   structure used to store algorithm state.
    Stp     -   initial estimate of the step length.
                Can be zero (no estimate).

  -- ALGLIB --
     Copyright 30.07.2010 by Bochkanov Sergey
*************************************************************************/
void mincgsuggeststep(const mincgstate &state, const double stp, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Modification of the preconditioner: preconditioning is turned off.

INPUT PARAMETERS:
    State   -   structure which stores algorithm state

NOTE:  you  can  change  preconditioner  "on  the  fly",  during algorithm
iterations.

  -- ALGLIB --
     Copyright 13.10.2010 by Bochkanov Sergey
*************************************************************************/
void mincgsetprecdefault(const mincgstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Modification  of  the  preconditioner:  diagonal of approximate Hessian is
used.

INPUT PARAMETERS:
    State   -   structure which stores algorithm state
    D       -   diagonal of the approximate Hessian, array[0..N-1],
                (if larger, only leading N elements are used).

NOTE:  you  can  change  preconditioner  "on  the  fly",  during algorithm
iterations.

NOTE 2: D[i] should be positive. Exception will be thrown otherwise.

NOTE 3: you should pass diagonal of approximate Hessian - NOT ITS INVERSE.

  -- ALGLIB --
     Copyright 13.10.2010 by Bochkanov Sergey
*************************************************************************/
void mincgsetprecdiag(const mincgstate &state, const real_1d_array &d, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Modification of the preconditioner: scale-based diagonal preconditioning.

This preconditioning mode can be useful when you  don't  have  approximate
diagonal of Hessian, but you know that your  variables  are  badly  scaled
(for  example,  one  variable is in [1,10], and another in [1000,100000]),
and most part of the ill-conditioning comes from different scales of vars.

In this case simple  scale-based  preconditioner,  with H[i] = 1/(s[i]^2),
can greatly improve convergence.

IMPRTANT: you should set scale of your variables with MinCGSetScale() call
(before or after MinCGSetPrecScale() call). Without knowledge of the scale
of your variables scale-based preconditioner will be just unit matrix.

INPUT PARAMETERS:
    State   -   structure which stores algorithm state

NOTE:  you  can  change  preconditioner  "on  the  fly",  during algorithm
iterations.

  -- ALGLIB --
     Copyright 13.10.2010 by Bochkanov Sergey
*************************************************************************/
void mincgsetprecscale(const mincgstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function provides reverse communication interface
Reverse communication interface is not documented or recommended to use.
See below for functions which provide better documented API
*************************************************************************/
bool mincgiteration(const mincgstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This family of functions is used to launcn iterations of nonlinear optimizer

These functions accept following parameters:
    state   -   algorithm state
    func    -   callback which calculates function (or merit function)
                value func at given point x
    grad    -   callback which calculates function (or merit function)
                value func and gradient grad at given point x
    rep     -   optional callback which is called after each iteration
                can be NULL
    ptr     -   optional pointer which is passed to func/grad/hess/jac/rep
                can be NULL

NOTES:

1. This function has two different implementations: one which  uses  exact
   (analytical) user-supplied  gradient, and one which uses function value
   only  and  numerically  differentiates  function  in  order  to  obtain
   gradient.

   Depending  on  the  specific  function  used to create optimizer object
   (either MinCGCreate()  for analytical gradient  or  MinCGCreateF()  for
   numerical differentiation) you should  choose  appropriate  variant  of
   MinCGOptimize() - one which accepts function AND gradient or one  which
   accepts function ONLY.

   Be careful to choose variant of MinCGOptimize()  which  corresponds  to
   your optimization scheme! Table below lists different  combinations  of
   callback (function/gradient) passed  to  MinCGOptimize()  and  specific
   function used to create optimizer.


                  |         USER PASSED TO MinCGOptimize()
   CREATED WITH   |  function only   |  function and gradient
   ------------------------------------------------------------
   MinCGCreateF() |     work                FAIL
   MinCGCreate()  |     FAIL                work

   Here "FAIL" denotes inappropriate combinations  of  optimizer  creation
   function and MinCGOptimize() version. Attemps to use  such  combination
   (for  example,  to create optimizer with  MinCGCreateF()  and  to  pass
   gradient information to MinCGOptimize()) will lead to  exception  being
   thrown. Either  you  did  not  pass  gradient when it WAS needed or you
   passed gradient when it was NOT needed.

  -- ALGLIB --
     Copyright 20.04.2009 by Bochkanov Sergey

*************************************************************************/
void mincgoptimize(mincgstate &state,
    void (*func)(const real_1d_array &x, double &func, void *ptr),
    void  (*rep)(const real_1d_array &x, double func, void *ptr) = NULL,
    void *ptr = NULL,
    const xparams _xparams = alglib::xdefault);
void mincgoptimize(mincgstate &state,
    void (*grad)(const real_1d_array &x, double &func, real_1d_array &grad, void *ptr),
    void  (*rep)(const real_1d_array &x, double func, void *ptr) = NULL,
    void *ptr = NULL,
    const xparams _xparams = alglib::xdefault);


/*************************************************************************
This  function  activates/deactivates verification  of  the  user-supplied
analytic gradient.

Upon  activation  of  this  option  OptGuard  integrity  checker  performs
numerical differentiation of your target function  at  the  initial  point
(note: future versions may also perform check  at  the  final  point)  and
compares numerical gradient with analytic one provided by you.

If difference is too large, an error flag is set and optimization  session
continues. After optimization session is over, you can retrieve the report
which  stores  both  gradients  and  specific  components  highlighted  as
suspicious by the OptGuard.

The primary OptGuard report can be retrieved with mincgoptguardresults().

IMPORTANT: gradient check is a high-overhead option which  will  cost  you
           about 3*N additional function evaluations. In many cases it may
           cost as much as the rest of the optimization session.

           YOU SHOULD NOT USE IT IN THE PRODUCTION CODE UNLESS YOU WANT TO
           CHECK DERIVATIVES PROVIDED BY SOME THIRD PARTY.

NOTE: unlike previous incarnation of the gradient checking code,  OptGuard
      does NOT interrupt optimization even if it discovers bad gradient.

INPUT PARAMETERS:
    State       -   structure used to store algorithm state
    TestStep    -   verification step used for numerical differentiation:
                    * TestStep=0 turns verification off
                    * TestStep>0 activates verification
                    You should carefully choose TestStep. Value  which  is
                    too large (so large that  function  behavior  is  non-
                    cubic at this scale) will lead  to  false  alarms. Too
                    short step will result in rounding  errors  dominating
                    numerical derivative.

                    You may use different step for different parameters by
                    means of setting scale with mincgsetscale().

=== EXPLANATION ==========================================================

In order to verify gradient algorithm performs following steps:
  * two trial steps are made to X[i]-TestStep*S[i] and X[i]+TestStep*S[i],
    where X[i] is i-th component of the initial point and S[i] is a  scale
    of i-th parameter
  * F(X) is evaluated at these trial points
  * we perform one more evaluation in the middle point of the interval
  * we  build  cubic  model using function values and derivatives at trial
    points and we compare its prediction with actual value in  the  middle
    point

  -- ALGLIB --
     Copyright 15.06.2014 by Bochkanov Sergey
*************************************************************************/
void mincgoptguardgradient(const mincgstate &state, const double teststep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This  function  activates/deactivates nonsmoothness monitoring  option  of
the  OptGuard  integrity  checker. Smoothness  monitor  silently  observes
solution process and tries to detect ill-posed problems, i.e. ones with:
a) discontinuous target function (non-C0)
b) nonsmooth     target function (non-C1)

Smoothness monitoring does NOT interrupt optimization  even if it suspects
that your problem is nonsmooth. It just sets corresponding  flags  in  the
OptGuard report which can be retrieved after optimization is over.

Smoothness monitoring is a moderate overhead option which often adds  less
than 1% to the optimizer running time. Thus, you can use it even for large
scale problems.

NOTE: OptGuard does  NOT  guarantee  that  it  will  always  detect  C0/C1
      continuity violations.

      First, minor errors are hard to  catch - say, a 0.0001 difference in
      the model values at two sides of the gap may be due to discontinuity
      of the model - or simply because the model has changed.

      Second, C1-violations  are  especially  difficult  to  detect  in  a
      noninvasive way. The optimizer usually  performs  very  short  steps
      near the nonsmoothness, and differentiation  usually   introduces  a
      lot of numerical noise.  It  is  hard  to  tell  whether  some  tiny
      discontinuity in the slope is due to real nonsmoothness or just  due
      to numerical noise alone.

      Our top priority was to avoid false positives, so in some rare cases
      minor errors may went unnoticed (however, in most cases they can  be
      spotted with restart from different initial point).

INPUT PARAMETERS:
    state   -   algorithm state
    level   -   monitoring level:
                * 0 - monitoring is disabled
                * 1 - noninvasive low-overhead monitoring; function values
                      and/or gradients are recorded, but OptGuard does not
                      try to perform additional evaluations  in  order  to
                      get more information about suspicious locations.

=== EXPLANATION ==========================================================

One major source of headache during optimization  is  the  possibility  of
the coding errors in the target function/constraints (or their gradients).
Such  errors   most   often   manifest   themselves  as  discontinuity  or
nonsmoothness of the target/constraints.

Another frequent situation is when you try to optimize something involving
lots of min() and max() operations, i.e. nonsmooth target. Although not  a
coding error, it is nonsmoothness anyway - and smooth  optimizers  usually
stop right after encountering nonsmoothness, well before reaching solution.

OptGuard integrity checker helps you to catch such situations: it monitors
function values/gradients being passed  to  the  optimizer  and  tries  to
errors. Upon discovering suspicious pair of points it  raises  appropriate
flag (and allows you to continue optimization). When optimization is done,
you can study OptGuard result.

  -- ALGLIB --
     Copyright 21.11.2018 by Bochkanov Sergey
*************************************************************************/
void mincgoptguardsmoothness(const mincgstate &state, const ae_int_t level, const xparams _xparams = alglib::xdefault);
void mincgoptguardsmoothness(const mincgstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Results of OptGuard integrity check, should be called  after  optimization
session is over.

=== PRIMARY REPORT =======================================================

OptGuard performs several checks which are intended to catch common errors
in the implementation of nonlinear function/gradient:
* incorrect analytic gradient
* discontinuous (non-C0) target functions (constraints)
* nonsmooth     (non-C1) target functions (constraints)

Each of these checks is activated with appropriate function:
* mincgoptguardgradient() for gradient verification
* mincgoptguardsmoothness() for C0/C1 checks

Following flags are set when these errors are suspected:
* rep.badgradsuspected, and additionally:
  * rep.badgradvidx for specific variable (gradient element) suspected
  * rep.badgradxbase, a point where gradient is tested
  * rep.badgraduser, user-provided gradient  (stored  as  2D  matrix  with
    single row in order to make  report  structure  compatible  with  more
    complex optimizers like MinNLC or MinLM)
  * rep.badgradnum,   reference    gradient    obtained    via   numerical
    differentiation (stored as  2D matrix with single row in order to make
    report structure compatible with more complex optimizers  like  MinNLC
    or MinLM)
* rep.nonc0suspected
* rep.nonc1suspected

=== ADDITIONAL REPORTS/LOGS ==============================================

Several different tests are performed to catch C0/C1 errors, you can  find
out specific test signaled error by looking to:
* rep.nonc0test0positive, for non-C0 test #0
* rep.nonc1test0positive, for non-C1 test #0
* rep.nonc1test1positive, for non-C1 test #1

Additional information (including line search logs)  can  be  obtained  by
means of:
* mincgoptguardnonc1test0results()
* mincgoptguardnonc1test1results()
which return detailed error reports, specific points where discontinuities
were found, and so on.

==========================================================================

INPUT PARAMETERS:
    state   -   algorithm state

OUTPUT PARAMETERS:
    rep     -   generic OptGuard report;  more  detailed  reports  can  be
                retrieved with other functions.

NOTE: false negatives (nonsmooth problems are not identified as  nonsmooth
      ones) are possible although unlikely.

      The reason  is  that  you  need  to  make several evaluations around
      nonsmoothness  in  order  to  accumulate  enough  information  about
      function curvature. Say, if you start right from the nonsmooth point,
      optimizer simply won't get enough data to understand what  is  going
      wrong before it terminates due to abrupt changes in the  derivative.
      It is also  possible  that  "unlucky"  step  will  move  us  to  the
      termination too quickly.

      Our current approach is to have less than 0.1%  false  negatives  in
      our test examples  (measured  with  multiple  restarts  from  random
      points), and to have exactly 0% false positives.

  -- ALGLIB --
     Copyright 21.11.2018 by Bochkanov Sergey
*************************************************************************/
void mincgoptguardresults(const mincgstate &state, optguardreport &rep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Detailed results of the OptGuard integrity check for nonsmoothness test #0

Nonsmoothness (non-C1) test #0 studies  function  values  (not  gradient!)
obtained during line searches and monitors  behavior  of  the  directional
derivative estimate.

This test is less powerful than test #1, but it does  not  depend  on  the
gradient values and thus it is more robust against artifacts introduced by
numerical differentiation.

Two reports are returned:
* a "strongest" one, corresponding  to  line   search  which  had  highest
  value of the nonsmoothness indicator
* a "longest" one, corresponding to line search which  had  more  function
  evaluations, and thus is more detailed

In both cases following fields are returned:

* positive - is TRUE  when test flagged suspicious point;  FALSE  if  test
  did not notice anything (in the latter cases fields below are empty).
* x0[], d[] - arrays of length N which store initial point  and  direction
  for line search (d[] can be normalized, but does not have to)
* stp[], f[] - arrays of length CNT which store step lengths and  function
  values at these points; f[i] is evaluated in x0+stp[i]*d.
* stpidxa, stpidxb - we  suspect  that  function  violates  C1  continuity
  between steps #stpidxa and #stpidxb (usually we have  stpidxb=stpidxa+3,
  with  most  likely  position  of  the  violation  between  stpidxa+1 and
  stpidxa+2.

==========================================================================
= SHORTLY SPEAKING: build a 2D plot of (stp,f) and look at it -  you  will
=                   see where C1 continuity is violated.
==========================================================================

INPUT PARAMETERS:
    state   -   algorithm state

OUTPUT PARAMETERS:
    strrep  -   C1 test #0 "strong" report
    lngrep  -   C1 test #0 "long" report

  -- ALGLIB --
     Copyright 21.11.2018 by Bochkanov Sergey
*************************************************************************/
void mincgoptguardnonc1test0results(const mincgstate &state, optguardnonc1test0report &strrep, optguardnonc1test0report &lngrep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Detailed results of the OptGuard integrity check for nonsmoothness test #1

Nonsmoothness (non-C1)  test  #1  studies  individual  components  of  the
gradient computed during line search.

When precise analytic gradient is provided this test is more powerful than
test #0  which  works  with  function  values  and  ignores  user-provided
gradient.  However,  test  #0  becomes  more   powerful   when   numerical
differentiation is employed (in such cases test #1 detects  higher  levels
of numerical noise and becomes too conservative).

This test also tells specific components of the gradient which violate  C1
continuity, which makes it more informative than #0, which just tells that
continuity is violated.

Two reports are returned:
* a "strongest" one, corresponding  to  line   search  which  had  highest
  value of the nonsmoothness indicator
* a "longest" one, corresponding to line search which  had  more  function
  evaluations, and thus is more detailed

In both cases following fields are returned:

* positive - is TRUE  when test flagged suspicious point;  FALSE  if  test
  did not notice anything (in the latter cases fields below are empty).
* vidx - is an index of the variable in [0,N) with nonsmooth derivative
* x0[], d[] - arrays of length N which store initial point  and  direction
  for line search (d[] can be normalized, but does not have to)
* stp[], g[] - arrays of length CNT which store step lengths and  gradient
  values at these points; g[i] is evaluated in  x0+stp[i]*d  and  contains
  vidx-th component of the gradient.
* stpidxa, stpidxb - we  suspect  that  function  violates  C1  continuity
  between steps #stpidxa and #stpidxb (usually we have  stpidxb=stpidxa+3,
  with  most  likely  position  of  the  violation  between  stpidxa+1 and
  stpidxa+2.

==========================================================================
= SHORTLY SPEAKING: build a 2D plot of (stp,f) and look at it -  you  will
=                   see where C1 continuity is violated.
==========================================================================

INPUT PARAMETERS:
    state   -   algorithm state

OUTPUT PARAMETERS:
    strrep  -   C1 test #1 "strong" report
    lngrep  -   C1 test #1 "long" report

  -- ALGLIB --
     Copyright 21.11.2018 by Bochkanov Sergey
*************************************************************************/
void mincgoptguardnonc1test1results(const mincgstate &state, optguardnonc1test1report &strrep, optguardnonc1test1report &lngrep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Conjugate gradient results

INPUT PARAMETERS:
    State   -   algorithm state

OUTPUT PARAMETERS:
    X       -   array[0..N-1], solution
    Rep     -   optimization report:
                * Rep.TerminationType completetion code:
                    * -8    internal integrity control  detected  infinite
                            or NAN values in  function/gradient.  Abnormal
                            termination signalled.
                    * -7    gradient verification failed.
                            See MinCGSetGradientCheck() for more information.
                    *  1    relative function improvement is no more than
                            EpsF.
                    *  2    relative step is no more than EpsX.
                    *  4    gradient norm is no more than EpsG
                    *  5    MaxIts steps was taken
                    *  7    stopping conditions are too stringent,
                            further improvement is impossible,
                            we return best X found so far
                    *  8    terminated by user
                * Rep.IterationsCount contains iterations count
                * NFEV countains number of function calculations

  -- ALGLIB --
     Copyright 20.04.2009 by Bochkanov Sergey
*************************************************************************/
void mincgresults(const mincgstate &state, real_1d_array &x, mincgreport &rep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Conjugate gradient results

Buffered implementation of MinCGResults(), which uses pre-allocated buffer
to store X[]. If buffer size is  too  small,  it  resizes  buffer.  It  is
intended to be used in the inner cycles of performance critical algorithms
where array reallocation penalty is too large to be ignored.

  -- ALGLIB --
     Copyright 20.04.2009 by Bochkanov Sergey
*************************************************************************/
void mincgresultsbuf(const mincgstate &state, real_1d_array &x, mincgreport &rep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This  subroutine  restarts  CG  algorithm from new point. All optimization
parameters are left unchanged.

This  function  allows  to  solve multiple  optimization  problems  (which
must have same number of dimensions) without object reallocation penalty.

INPUT PARAMETERS:
    State   -   structure used to store algorithm state.
    X       -   new starting point.

  -- ALGLIB --
     Copyright 30.07.2010 by Bochkanov Sergey
*************************************************************************/
void mincgrestartfrom(const mincgstate &state, const real_1d_array &x, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This subroutine submits request for termination of running  optimizer.  It
should be called from user-supplied callback when user decides that it  is
time to "smoothly" terminate optimization process.  As  result,  optimizer
stops at point which was "current accepted" when termination  request  was
submitted and returns error code 8 (successful termination).

INPUT PARAMETERS:
    State   -   optimizer structure

NOTE: after  request  for  termination  optimizer  may   perform   several
      additional calls to user-supplied callbacks. It does  NOT  guarantee
      to stop immediately - it just guarantees that these additional calls
      will be discarded later.

NOTE: calling this function on optimizer which is NOT running will have no
      effect.

NOTE: multiple calls to this function are possible. First call is counted,
      subsequent calls are silently ignored.

  -- ALGLIB --
     Copyright 08.10.2014 by Bochkanov Sergey
*************************************************************************/
void mincgrequesttermination(const mincgstate &state, const xparams _xparams = alglib::xdefault);
#endif

#if defined(AE_COMPILE_MINLM) || !defined(AE_PARTIAL_BUILD)
/*************************************************************************
                IMPROVED LEVENBERG-MARQUARDT METHOD FOR
                 NON-LINEAR LEAST SQUARES OPTIMIZATION

DESCRIPTION:
This function is used to find minimum of function which is represented  as
sum of squares:
    F(x) = f[0]^2(x[0],...,x[n-1]) + ... + f[m-1]^2(x[0],...,x[n-1])
using value of function vector f[] and Jacobian of f[].


REQUIREMENTS:
This algorithm will request following information during its operation:

* function vector f[] at given point X
* function vector f[] and Jacobian of f[] (simultaneously) at given point

There are several overloaded versions of  MinLMOptimize()  function  which
correspond  to  different LM-like optimization algorithms provided by this
unit. You should choose version which accepts fvec()  and jac() callbacks.
First  one  is used to calculate f[] at given point, second one calculates
f[] and Jacobian df[i]/dx[j].

You can try to initialize MinLMState structure with VJ  function and  then
use incorrect version  of  MinLMOptimize()  (for  example,  version  which
works  with  general  form function and does not provide Jacobian), but it
will  lead  to  exception  being  thrown  after first attempt to calculate
Jacobian.


USAGE:
1. User initializes algorithm state with MinLMCreateVJ() call
2. User tunes solver parameters with MinLMSetCond(),  MinLMSetStpMax() and
   other functions
3. User calls MinLMOptimize() function which  takes algorithm  state   and
   callback functions.
4. User calls MinLMResults() to get solution
5. Optionally, user may call MinLMRestartFrom() to solve  another  problem
   with same N/M but another starting point and/or another function.
   MinLMRestartFrom() allows to reuse already initialized structure.


INPUT PARAMETERS:
    N       -   dimension, N>1
                * if given, only leading N elements of X are used
                * if not given, automatically determined from size of X
    M       -   number of functions f[i]
    X       -   initial solution, array[0..N-1]

OUTPUT PARAMETERS:
    State   -   structure which stores algorithm state

NOTES:
1. you may tune stopping conditions with MinLMSetCond() function
2. if target function contains exp() or other fast growing functions,  and
   optimization algorithm makes too large steps which leads  to  overflow,
   use MinLMSetStpMax() function to bound algorithm's steps.

  -- ALGLIB --
     Copyright 30.03.2009 by Bochkanov Sergey
*************************************************************************/
void minlmcreatevj(const ae_int_t n, const ae_int_t m, const real_1d_array &x, minlmstate &state, const xparams _xparams = alglib::xdefault);
void minlmcreatevj(const ae_int_t m, const real_1d_array &x, minlmstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
                IMPROVED LEVENBERG-MARQUARDT METHOD FOR
                 NON-LINEAR LEAST SQUARES OPTIMIZATION

DESCRIPTION:
This function is used to find minimum of function which is represented  as
sum of squares:
    F(x) = f[0]^2(x[0],...,x[n-1]) + ... + f[m-1]^2(x[0],...,x[n-1])
using value of function vector f[] only. Finite differences  are  used  to
calculate Jacobian.


REQUIREMENTS:
This algorithm will request following information during its operation:
* function vector f[] at given point X

There are several overloaded versions of  MinLMOptimize()  function  which
correspond  to  different LM-like optimization algorithms provided by this
unit. You should choose version which accepts fvec() callback.

You can try to initialize MinLMState structure with VJ  function and  then
use incorrect version  of  MinLMOptimize()  (for  example,  version  which
works with general form function and does not accept function vector), but
it will  lead  to  exception being thrown after first attempt to calculate
Jacobian.


USAGE:
1. User initializes algorithm state with MinLMCreateV() call
2. User tunes solver parameters with MinLMSetCond(),  MinLMSetStpMax() and
   other functions
3. User calls MinLMOptimize() function which  takes algorithm  state   and
   callback functions.
4. User calls MinLMResults() to get solution
5. Optionally, user may call MinLMRestartFrom() to solve  another  problem
   with same N/M but another starting point and/or another function.
   MinLMRestartFrom() allows to reuse already initialized structure.


INPUT PARAMETERS:
    N       -   dimension, N>1
                * if given, only leading N elements of X are used
                * if not given, automatically determined from size of X
    M       -   number of functions f[i]
    X       -   initial solution, array[0..N-1]
    DiffStep-   differentiation step, >0

OUTPUT PARAMETERS:
    State   -   structure which stores algorithm state

See also MinLMIteration, MinLMResults.

NOTES:
1. you may tune stopping conditions with MinLMSetCond() function
2. if target function contains exp() or other fast growing functions,  and
   optimization algorithm makes too large steps which leads  to  overflow,
   use MinLMSetStpMax() function to bound algorithm's steps.

  -- ALGLIB --
     Copyright 30.03.2009 by Bochkanov Sergey
*************************************************************************/
void minlmcreatev(const ae_int_t n, const ae_int_t m, const real_1d_array &x, const double diffstep, minlmstate &state, const xparams _xparams = alglib::xdefault);
void minlmcreatev(const ae_int_t m, const real_1d_array &x, const double diffstep, minlmstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
    LEVENBERG-MARQUARDT-LIKE METHOD FOR NON-LINEAR OPTIMIZATION

DESCRIPTION:
This  function  is  used  to  find  minimum  of general form (not "sum-of-
-squares") function
    F = F(x[0], ..., x[n-1])
using  its  gradient  and  Hessian.  Levenberg-Marquardt modification with
L-BFGS pre-optimization and internal pre-conditioned  L-BFGS  optimization
after each Levenberg-Marquardt step is used.


REQUIREMENTS:
This algorithm will request following information during its operation:

* function value F at given point X
* F and gradient G (simultaneously) at given point X
* F, G and Hessian H (simultaneously) at given point X

There are several overloaded versions of  MinLMOptimize()  function  which
correspond  to  different LM-like optimization algorithms provided by this
unit. You should choose version which accepts func(),  grad()  and  hess()
function pointers. First pointer is used to calculate F  at  given  point,
second  one  calculates  F(x)  and  grad F(x),  third one calculates F(x),
grad F(x), hess F(x).

You can try to initialize MinLMState structure with FGH-function and  then
use incorrect version of MinLMOptimize() (for example, version which  does
not provide Hessian matrix), but it will lead to  exception  being  thrown
after first attempt to calculate Hessian.


USAGE:
1. User initializes algorithm state with MinLMCreateFGH() call
2. User tunes solver parameters with MinLMSetCond(),  MinLMSetStpMax() and
   other functions
3. User calls MinLMOptimize() function which  takes algorithm  state   and
   pointers (delegates, etc.) to callback functions.
4. User calls MinLMResults() to get solution
5. Optionally, user may call MinLMRestartFrom() to solve  another  problem
   with same N but another starting point and/or another function.
   MinLMRestartFrom() allows to reuse already initialized structure.


INPUT PARAMETERS:
    N       -   dimension, N>1
                * if given, only leading N elements of X are used
                * if not given, automatically determined from size of X
    X       -   initial solution, array[0..N-1]

OUTPUT PARAMETERS:
    State   -   structure which stores algorithm state

NOTES:
1. you may tune stopping conditions with MinLMSetCond() function
2. if target function contains exp() or other fast growing functions,  and
   optimization algorithm makes too large steps which leads  to  overflow,
   use MinLMSetStpMax() function to bound algorithm's steps.

  -- ALGLIB --
     Copyright 30.03.2009 by Bochkanov Sergey
*************************************************************************/
void minlmcreatefgh(const ae_int_t n, const real_1d_array &x, minlmstate &state, const xparams _xparams = alglib::xdefault);
void minlmcreatefgh(const real_1d_array &x, minlmstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets stopping conditions for Levenberg-Marquardt optimization
algorithm.

INPUT PARAMETERS:
    State   -   structure which stores algorithm state
    EpsX    -   >=0
                The subroutine finishes its work if  on  k+1-th  iteration
                the condition |v|<=EpsX is fulfilled, where:
                * |.| means Euclidian norm
                * v - scaled step vector, v[i]=dx[i]/s[i]
                * dx - ste pvector, dx=X(k+1)-X(k)
                * s - scaling coefficients set by MinLMSetScale()
                Recommended values: 1E-9 ... 1E-12.
    MaxIts  -   maximum number of iterations. If MaxIts=0, the  number  of
                iterations   is    unlimited.   Only   Levenberg-Marquardt
                iterations  are  counted  (L-BFGS/CG  iterations  are  NOT
                counted because their cost is very low compared to that of
                LM).

Passing  EpsX=0  and  MaxIts=0  (simultaneously)  will  lead  to automatic
stopping criterion selection (small EpsX).

NOTE: it is not recommended to set large EpsX (say, 0.001). Because LM  is
      a second-order method, it performs very precise steps anyway.

  -- ALGLIB --
     Copyright 02.04.2010 by Bochkanov Sergey
*************************************************************************/
void minlmsetcond(const minlmstate &state, const double epsx, const ae_int_t maxits, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function turns on/off reporting.

INPUT PARAMETERS:
    State   -   structure which stores algorithm state
    NeedXRep-   whether iteration reports are needed or not

If NeedXRep is True, algorithm will call rep() callback function if  it is
provided to MinLMOptimize(). Both Levenberg-Marquardt and internal  L-BFGS
iterations are reported.

  -- ALGLIB --
     Copyright 02.04.2010 by Bochkanov Sergey
*************************************************************************/
void minlmsetxrep(const minlmstate &state, const bool needxrep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets maximum step length

INPUT PARAMETERS:
    State   -   structure which stores algorithm state
    StpMax  -   maximum step length, >=0. Set StpMax to 0.0,  if you don't
                want to limit step length.

Use this subroutine when you optimize target function which contains exp()
or  other  fast  growing  functions,  and optimization algorithm makes too
large  steps  which  leads  to overflow. This function allows us to reject
steps  that  are  too  large  (and  therefore  expose  us  to the possible
overflow) without actually calculating function value at the x+stp*d.

NOTE: non-zero StpMax leads to moderate  performance  degradation  because
intermediate  step  of  preconditioned L-BFGS optimization is incompatible
with limits on step size.

  -- ALGLIB --
     Copyright 02.04.2010 by Bochkanov Sergey
*************************************************************************/
void minlmsetstpmax(const minlmstate &state, const double stpmax, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets scaling coefficients for LM optimizer.

ALGLIB optimizers use scaling matrices to test stopping  conditions  (step
size and gradient are scaled before comparison with tolerances).  Scale of
the I-th variable is a translation invariant measure of:
a) "how large" the variable is
b) how large the step should be to make significant changes in the function

Generally, scale is NOT considered to be a form of preconditioner.  But LM
optimizer is unique in that it uses scaling matrix both  in  the  stopping
condition tests and as Marquardt damping factor.

Proper scaling is very important for the algorithm performance. It is less
important for the quality of results, but still has some influence (it  is
easier  to  converge  when  variables  are  properly  scaled, so premature
stopping is possible when very badly scalled variables are  combined  with
relaxed stopping conditions).

INPUT PARAMETERS:
    State   -   structure stores algorithm state
    S       -   array[N], non-zero scaling coefficients
                S[i] may be negative, sign doesn't matter.

  -- ALGLIB --
     Copyright 14.01.2011 by Bochkanov Sergey
*************************************************************************/
void minlmsetscale(const minlmstate &state, const real_1d_array &s, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets boundary constraints for LM optimizer

Boundary constraints are inactive by default (after initial creation).
They are preserved until explicitly turned off with another SetBC() call.

INPUT PARAMETERS:
    State   -   structure stores algorithm state
    BndL    -   lower bounds, array[N].
                If some (all) variables are unbounded, you may specify
                very small number or -INF (latter is recommended because
                it will allow solver to use better algorithm).
    BndU    -   upper bounds, array[N].
                If some (all) variables are unbounded, you may specify
                very large number or +INF (latter is recommended because
                it will allow solver to use better algorithm).

NOTE 1: it is possible to specify BndL[i]=BndU[i]. In this case I-th
variable will be "frozen" at X[i]=BndL[i]=BndU[i].

NOTE 2: this solver has following useful properties:
* bound constraints are always satisfied exactly
* function is evaluated only INSIDE area specified by bound constraints
  or at its boundary

  -- ALGLIB --
     Copyright 14.01.2011 by Bochkanov Sergey
*************************************************************************/
void minlmsetbc(const minlmstate &state, const real_1d_array &bndl, const real_1d_array &bndu, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function sets general linear constraints for LM optimizer

Linear constraints are inactive by default (after initial creation).  They
are preserved until explicitly turned off with another minlmsetlc() call.

INPUT PARAMETERS:
    State   -   structure stores algorithm state
    C       -   linear constraints, array[K,N+1].
                Each row of C represents one constraint, either equality
                or inequality (see below):
                * first N elements correspond to coefficients,
                * last element corresponds to the right part.
                All elements of C (including right part) must be finite.
    CT      -   type of constraints, array[K]:
                * if CT[i]>0, then I-th constraint is C[i,*]*x >= C[i,n+1]
                * if CT[i]=0, then I-th constraint is C[i,*]*x  = C[i,n+1]
                * if CT[i]<0, then I-th constraint is C[i,*]*x <= C[i,n+1]
    K       -   number of equality/inequality constraints, K>=0:
                * if given, only leading K elements of C/CT are used
                * if not given, automatically determined from sizes of C/CT

IMPORTANT: if you have linear constraints, it is strongly  recommended  to
           set scale of variables with minlmsetscale(). QP solver which is
           used to calculate linearly constrained steps heavily relies  on
           good scaling of input problems.

IMPORTANT: solvers created with minlmcreatefgh()  do  not  support  linear
           constraints.

NOTE: linear  (non-bound)  constraints are satisfied only approximately  -
      there  always  exists some violation due  to  numerical  errors  and
      algorithmic limitations.

NOTE: general linear constraints  add  significant  overhead  to  solution
      process. Although solver performs roughly same amount of  iterations
      (when compared  with  similar  box-only  constrained  problem), each
      iteration   now    involves  solution  of  linearly  constrained  QP
      subproblem, which requires ~3-5 times more Cholesky  decompositions.
      Thus, if you can reformulate your problem in such way  this  it  has
      only box constraints, it may be beneficial to do so.

  -- ALGLIB --
     Copyright 14.01.2011 by Bochkanov Sergey
*************************************************************************/
void minlmsetlc(const minlmstate &state, const real_2d_array &c, const integer_1d_array &ct, const ae_int_t k, const xparams _xparams = alglib::xdefault);
void minlmsetlc(const minlmstate &state, const real_2d_array &c, const integer_1d_array &ct, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function is used to change acceleration settings

You can choose between three acceleration strategies:
* AccType=0, no acceleration.
* AccType=1, secant updates are used to update quadratic model after  each
  iteration. After fixed number of iterations (or after  model  breakdown)
  we  recalculate  quadratic  model  using  analytic  Jacobian  or  finite
  differences. Number of secant-based iterations depends  on  optimization
  settings: about 3 iterations - when we have analytic Jacobian, up to 2*N
  iterations - when we use finite differences to calculate Jacobian.

AccType=1 is recommended when Jacobian  calculation  cost is prohibitively
high (several Mx1 function vector calculations  followed  by  several  NxN
Cholesky factorizations are faster than calculation of one M*N  Jacobian).
It should also be used when we have no Jacobian, because finite difference
approximation takes too much time to compute.

Table below list  optimization  protocols  (XYZ  protocol  corresponds  to
MinLMCreateXYZ) and acceleration types they support (and use by  default).

ACCELERATION TYPES SUPPORTED BY OPTIMIZATION PROTOCOLS:

protocol    0   1   comment
V           +   +
VJ          +   +
FGH         +

DEFAULT VALUES:

protocol    0   1   comment
V               x   without acceleration it is so slooooooooow
VJ          x
FGH         x

NOTE: this  function should be called before optimization. Attempt to call
it during algorithm iterations may result in unexpected behavior.

NOTE: attempt to call this function with unsupported protocol/acceleration
combination will result in exception being thrown.

  -- ALGLIB --
     Copyright 14.10.2010 by Bochkanov Sergey
*************************************************************************/
void minlmsetacctype(const minlmstate &state, const ae_int_t acctype, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function provides reverse communication interface
Reverse communication interface is not documented or recommended to use.
See below for functions which provide better documented API
*************************************************************************/
bool minlmiteration(const minlmstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This family of functions is used to launcn iterations of nonlinear optimizer

These functions accept following parameters:
    state   -   algorithm state
    func    -   callback which calculates function (or merit function)
                value func at given point x
    grad    -   callback which calculates function (or merit function)
                value func and gradient grad at given point x
    hess    -   callback which calculates function (or merit function)
                value func, gradient grad and Hessian hess at given point x
    fvec    -   callback which calculates function vector fi[]
                at given point x
    jac     -   callback which calculates function vector fi[]
                and Jacobian jac at given point x
    rep     -   optional callback which is called after each iteration
                can be NULL
    ptr     -   optional pointer which is passed to func/grad/hess/jac/rep
                can be NULL

NOTES:

1. Depending on function used to create state  structure,  this  algorithm
   may accept Jacobian and/or Hessian and/or gradient.  According  to  the
   said above, there ase several versions of this function,  which  accept
   different sets of callbacks.

   This flexibility opens way to subtle errors - you may create state with
   MinLMCreateFGH() (optimization using Hessian), but call function  which
   does not accept Hessian. So when algorithm will request Hessian,  there
   will be no callback to call. In this case exception will be thrown.

   Be careful to avoid such errors because there is no way to find them at
   compile time - you can see them at runtime only.

  -- ALGLIB --
     Copyright 10.03.2009 by Bochkanov Sergey

*************************************************************************/
void minlmoptimize(minlmstate &state,
    void (*fvec)(const real_1d_array &x, real_1d_array &fi, void *ptr),
    void  (*rep)(const real_1d_array &x, double func, void *ptr) = NULL,
    void *ptr = NULL,
    const xparams _xparams = alglib::xdefault);
void minlmoptimize(minlmstate &state,
    void (*fvec)(const real_1d_array &x, real_1d_array &fi, void *ptr),
    void  (*jac)(const real_1d_array &x, real_1d_array &fi, real_2d_array &jac, void *ptr),
    void  (*rep)(const real_1d_array &x, double func, void *ptr) = NULL,
    void *ptr = NULL,
    const xparams _xparams = alglib::xdefault);
void minlmoptimize(minlmstate &state,
    void (*func)(const real_1d_array &x, double &func, void *ptr),
    void (*grad)(const real_1d_array &x, double &func, real_1d_array &grad, void *ptr),
    void (*hess)(const real_1d_array &x, double &func, real_1d_array &grad, real_2d_array &hess, void *ptr),
    void  (*rep)(const real_1d_array &x, double func, void *ptr) = NULL,
    void *ptr = NULL,
    const xparams _xparams = alglib::xdefault);
void minlmoptimize(minlmstate &state,
    void (*func)(const real_1d_array &x, double &func, void *ptr),
    void  (*jac)(const real_1d_array &x, real_1d_array &fi, real_2d_array &jac, void *ptr),
    void  (*rep)(const real_1d_array &x, double func, void *ptr) = NULL,
    void *ptr = NULL,
    const xparams _xparams = alglib::xdefault);
void minlmoptimize(minlmstate &state,
    void (*func)(const real_1d_array &x, double &func, void *ptr),
    void (*grad)(const real_1d_array &x, double &func, real_1d_array &grad, void *ptr),
    void  (*jac)(const real_1d_array &x, real_1d_array &fi, real_2d_array &jac, void *ptr),
    void  (*rep)(const real_1d_array &x, double func, void *ptr) = NULL,
    void *ptr = NULL,
    const xparams _xparams = alglib::xdefault);


/*************************************************************************
This  function  activates/deactivates verification  of  the  user-supplied
analytic Jacobian.

Upon  activation  of  this  option  OptGuard  integrity  checker  performs
numerical differentiation of your target function vector  at  the  initial
point (note: future versions may also perform check  at  the final  point)
and compares numerical Jacobian with analytic one provided by you.

If difference is too large, an error flag is set and optimization  session
continues. After optimization session is over, you can retrieve the report
which stores  both  Jacobians,  and  specific  components  highlighted  as
suspicious by the OptGuard.

The OptGuard report can be retrieved with minlmoptguardresults().

IMPORTANT: gradient check is a high-overhead option which  will  cost  you
           about 3*N additional function evaluations. In many cases it may
           cost as much as the rest of the optimization session.

           YOU SHOULD NOT USE IT IN THE PRODUCTION CODE UNLESS YOU WANT TO
           CHECK DERIVATIVES PROVIDED BY SOME THIRD PARTY.

NOTE: unlike previous incarnation of the gradient checking code,  OptGuard
      does NOT interrupt optimization even if it discovers bad gradient.

INPUT PARAMETERS:
    State       -   structure used to store algorithm state
    TestStep    -   verification step used for numerical differentiation:
                    * TestStep=0 turns verification off
                    * TestStep>0 activates verification
                    You should carefully choose TestStep. Value  which  is
                    too large (so large that  function  behavior  is  non-
                    cubic at this scale) will lead  to  false  alarms. Too
                    short step will result in rounding  errors  dominating
                    numerical derivative.

                    You may use different step for different parameters by
                    means of setting scale with minlmsetscale().

=== EXPLANATION ==========================================================

In order to verify gradient algorithm performs following steps:
  * two trial steps are made to X[i]-TestStep*S[i] and X[i]+TestStep*S[i],
    where X[i] is i-th component of the initial point and S[i] is a  scale
    of i-th parameter
  * F(X) is evaluated at these trial points
  * we perform one more evaluation in the middle point of the interval
  * we  build  cubic  model using function values and derivatives at trial
    points and we compare its prediction with actual value in  the  middle
    point

  -- ALGLIB --
     Copyright 15.06.2014 by Bochkanov Sergey
*************************************************************************/
void minlmoptguardgradient(const minlmstate &state, const double teststep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Results of OptGuard integrity check, should be called  after  optimization
session is over.

OptGuard checks analytic Jacobian  against  reference  value  obtained  by
numerical differentiation with user-specified step.

NOTE: other optimizers perform additional OptGuard checks for things  like
      C0/C1-continuity violations. However, LM optimizer  can  check  only
      for incorrect Jacobian.

      The reason is that unlike line search methods LM optimizer does  not
      perform extensive evaluations along the line. Thus, we simply do not
      have enough data to catch C0/C1-violations.

This check is activated with  minlmoptguardgradient() function.

Following flags are set when these errors are suspected:
* rep.badgradsuspected, and additionally:
  * rep.badgradfidx for specific function (Jacobian row) suspected
  * rep.badgradvidx for specific variable (Jacobian column) suspected
  * rep.badgradxbase, a point where gradient/Jacobian is tested
  * rep.badgraduser, user-provided gradient/Jacobian
  * rep.badgradnum, reference gradient/Jacobian obtained via numerical
    differentiation

INPUT PARAMETERS:
    state   -   algorithm state

OUTPUT PARAMETERS:
    rep     -   OptGuard report

  -- ALGLIB --
     Copyright 21.11.2018 by Bochkanov Sergey
*************************************************************************/
void minlmoptguardresults(const minlmstate &state, optguardreport &rep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Levenberg-Marquardt algorithm results

NOTE: if you activated OptGuard integrity checking functionality and  want
      to get OptGuard report,  it  can  be  retrieved  with  the  help  of
      minlmoptguardresults() function.

INPUT PARAMETERS:
    State   -   algorithm state

OUTPUT PARAMETERS:
    X       -   array[0..N-1], solution
    Rep     -   optimization  report;  includes  termination   codes   and
                additional information. Termination codes are listed below,
                see comments for this structure for more info.
                Termination code is stored in rep.terminationtype field:
                * -8    optimizer detected NAN/INF values either in the
                        function itself, or in its Jacobian
                * -3    constraints are inconsistent
                *  2    relative step is no more than EpsX.
                *  5    MaxIts steps was taken
                *  7    stopping conditions are too stringent,
                        further improvement is impossible
                *  8    terminated by user who called minlmrequesttermination().
                        X contains point which was "current accepted" when
                        termination request was submitted.

  -- ALGLIB --
     Copyright 10.03.2009 by Bochkanov Sergey
*************************************************************************/
void minlmresults(const minlmstate &state, real_1d_array &x, minlmreport &rep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
Levenberg-Marquardt algorithm results

Buffered implementation of MinLMResults(), which uses pre-allocated buffer
to store X[]. If buffer size is  too  small,  it  resizes  buffer.  It  is
intended to be used in the inner cycles of performance critical algorithms
where array reallocation penalty is too large to be ignored.

  -- ALGLIB --
     Copyright 10.03.2009 by Bochkanov Sergey
*************************************************************************/
void minlmresultsbuf(const minlmstate &state, real_1d_array &x, minlmreport &rep, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This  subroutine  restarts  LM  algorithm from new point. All optimization
parameters are left unchanged.

This  function  allows  to  solve multiple  optimization  problems  (which
must have same number of dimensions) without object reallocation penalty.

INPUT PARAMETERS:
    State   -   structure used for reverse communication previously
                allocated with MinLMCreateXXX call.
    X       -   new starting point.

  -- ALGLIB --
     Copyright 30.07.2010 by Bochkanov Sergey
*************************************************************************/
void minlmrestartfrom(const minlmstate &state, const real_1d_array &x, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This subroutine submits request for termination of running  optimizer.  It
should be called from user-supplied callback when user decides that it  is
time to "smoothly" terminate optimization process.  As  result,  optimizer
stops at point which was "current accepted" when termination  request  was
submitted and returns error code 8 (successful termination).

INPUT PARAMETERS:
    State   -   optimizer structure

NOTE: after  request  for  termination  optimizer  may   perform   several
      additional calls to user-supplied callbacks. It does  NOT  guarantee
      to stop immediately - it just guarantees that these additional calls
      will be discarded later.

NOTE: calling this function on optimizer which is NOT running will have no
      effect.

NOTE: multiple calls to this function are possible. First call is counted,
      subsequent calls are silently ignored.

  -- ALGLIB --
     Copyright 08.10.2014 by Bochkanov Sergey
*************************************************************************/
void minlmrequesttermination(const minlmstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This is obsolete function.

Since ALGLIB 3.3 it is equivalent to MinLMCreateVJ().

  -- ALGLIB --
     Copyright 30.03.2009 by Bochkanov Sergey
*************************************************************************/
void minlmcreatevgj(const ae_int_t n, const ae_int_t m, const real_1d_array &x, minlmstate &state, const xparams _xparams = alglib::xdefault);
void minlmcreatevgj(const ae_int_t m, const real_1d_array &x, minlmstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This is obsolete function.

Since ALGLIB 3.3 it is equivalent to MinLMCreateFJ().

  -- ALGLIB --
     Copyright 30.03.2009 by Bochkanov Sergey
*************************************************************************/
void minlmcreatefgj(const ae_int_t n, const ae_int_t m, const real_1d_array &x, minlmstate &state, const xparams _xparams = alglib::xdefault);
void minlmcreatefgj(const ae_int_t m, const real_1d_array &x, minlmstate &state, const xparams _xparams = alglib::xdefault);


/*************************************************************************
This function is considered obsolete since ALGLIB 3.1.0 and is present for
backward  compatibility  only.  We  recommend  to use MinLMCreateVJ, which
provides similar, but more consistent and feature-rich interface.

  -- ALGLIB --
     Copyright 30.03.2009 by Bochkanov Sergey
*************************************************************************/
void minlmcreatefj(const ae_int_t n, const ae_int_t m, const real_1d_array &x, minlmstate &state, const xparams _xparams = alglib::xdefault);
void minlmcreatefj(const ae_int_t m, const real_1d_array &x, minlmstate &state, const xparams _xparams = alglib::xdefault);
#endif
}

/////////////////////////////////////////////////////////////////////////
//
// THIS SECTION CONTAINS COMPUTATIONAL CORE DECLARATIONS (FUNCTIONS)
//
/////////////////////////////////////////////////////////////////////////
namespace alglib_impl
{
#if defined(AE_COMPILE_CQMODELS) || !defined(AE_PARTIAL_BUILD)
void cqminit(ae_int_t n, convexquadraticmodel* s, ae_state *_state);
void cqmseta(convexquadraticmodel* s,
     /* Real    */ ae_matrix* a,
     ae_bool isupper,
     double alpha,
     ae_state *_state);
void cqmgeta(convexquadraticmodel* s,
     /* Real    */ ae_matrix* a,
     ae_state *_state);
void cqmrewritedensediagonal(convexquadraticmodel* s,
     /* Real    */ ae_vector* z,
     ae_state *_state);
void cqmsetd(convexquadraticmodel* s,
     /* Real    */ ae_vector* d,
     double tau,
     ae_state *_state);
void cqmdropa(convexquadraticmodel* s, ae_state *_state);
void cqmsetb(convexquadraticmodel* s,
     /* Real    */ ae_vector* b,
     ae_state *_state);
void cqmsetq(convexquadraticmodel* s,
     /* Real    */ ae_matrix* q,
     /* Real    */ ae_vector* r,
     ae_int_t k,
     double theta,
     ae_state *_state);
void cqmsetactiveset(convexquadraticmodel* s,
     /* Real    */ ae_vector* x,
     /* Boolean */ ae_vector* activeset,
     ae_state *_state);
double cqmeval(convexquadraticmodel* s,
     /* Real    */ ae_vector* x,
     ae_state *_state);
void cqmevalx(convexquadraticmodel* s,
     /* Real    */ ae_vector* x,
     double* r,
     double* noise,
     ae_state *_state);
void cqmgradunconstrained(convexquadraticmodel* s,
     /* Real    */ ae_vector* x,
     /* Real    */ ae_vector* g,
     ae_state *_state);
double cqmxtadx2(convexquadraticmodel* s,
     /* Real    */ ae_vector* x,
     /* Real    */ ae_vector* tmp,
     ae_state *_state);
void cqmadx(convexquadraticmodel* s,
     /* Real    */ ae_vector* x,
     /* Real    */ ae_vector* y,
     ae_state *_state);
ae_bool cqmconstrainedoptimum(convexquadraticmodel* s,
     /* Real    */ ae_vector* x,
     ae_state *_state);
void cqmscalevector(convexquadraticmodel* s,
     /* Real    */ ae_vector* x,
     ae_state *_state);
void cqmgetdiaga(convexquadraticmodel* s,
     /* Real    */ ae_vector* x,
     ae_state *_state);
double cqmdebugconstrainedevalt(convexquadraticmodel* s,
     /* Real    */ ae_vector* x,
     ae_state *_state);
double cqmdebugconstrainedevale(convexquadraticmodel* s,
     /* Real    */ ae_vector* x,
     ae_state *_state);
void _convexquadraticmodel_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _convexquadraticmodel_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _convexquadraticmodel_clear(void* _p);
void _convexquadraticmodel_destroy(void* _p);
#endif
#if defined(AE_COMPILE_OPTGUARDAPI) || !defined(AE_PARTIAL_BUILD)
void optguardinitinternal(optguardreport* rep,
     ae_int_t n,
     ae_int_t k,
     ae_state *_state);
void optguardexportreport(optguardreport* srcrep,
     ae_int_t n,
     ae_int_t k,
     ae_bool badgradhasxj,
     optguardreport* dstrep,
     ae_state *_state);
void smoothnessmonitorexportc1test0report(optguardnonc1test0report* srcrep,
     /* Real    */ ae_vector* s,
     optguardnonc1test0report* dstrep,
     ae_state *_state);
void smoothnessmonitorexportc1test1report(optguardnonc1test1report* srcrep,
     /* Real    */ ae_vector* s,
     optguardnonc1test1report* dstrep,
     ae_state *_state);
ae_bool optguardallclear(optguardreport* rep, ae_state *_state);
void _optguardreport_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _optguardreport_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _optguardreport_clear(void* _p);
void _optguardreport_destroy(void* _p);
void _optguardnonc1test0report_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _optguardnonc1test0report_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _optguardnonc1test0report_clear(void* _p);
void _optguardnonc1test0report_destroy(void* _p);
void _optguardnonc1test1report_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _optguardnonc1test1report_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _optguardnonc1test1report_clear(void* _p);
void _optguardnonc1test1report_destroy(void* _p);
#endif
#if defined(AE_COMPILE_OPTSERV) || !defined(AE_PARTIAL_BUILD)
void checkbcviolation(/* Boolean */ ae_vector* hasbndl,
     /* Real    */ ae_vector* bndl,
     /* Boolean */ ae_vector* hasbndu,
     /* Real    */ ae_vector* bndu,
     /* Real    */ ae_vector* x,
     ae_int_t n,
     /* Real    */ ae_vector* s,
     ae_bool nonunits,
     double* bcerr,
     ae_int_t* bcidx,
     ae_state *_state);
void checklcviolation(/* Real    */ ae_matrix* cleic,
     /* Integer */ ae_vector* lcsrcidx,
     ae_int_t nec,
     ae_int_t nic,
     /* Real    */ ae_vector* x,
     ae_int_t n,
     double* lcerr,
     ae_int_t* lcidx,
     ae_state *_state);
void checknlcviolation(/* Real    */ ae_vector* fi,
     ae_int_t ng,
     ae_int_t nh,
     double* nlcerr,
     ae_int_t* nlcidx,
     ae_state *_state);
void trimprepare(double f, double* threshold, ae_state *_state);
void trimfunction(double* f,
     /* Real    */ ae_vector* g,
     ae_int_t n,
     double threshold,
     ae_state *_state);
ae_bool enforceboundaryconstraints(/* Real    */ ae_vector* x,
     /* Real    */ ae_vector* bl,
     /* Boolean */ ae_vector* havebl,
     /* Real    */ ae_vector* bu,
     /* Boolean */ ae_vector* havebu,
     ae_int_t nmain,
     ae_int_t nslack,
     ae_state *_state);
void projectgradientintobc(/* Real    */ ae_vector* x,
     /* Real    */ ae_vector* g,
     /* Real    */ ae_vector* bl,
     /* Boolean */ ae_vector* havebl,
     /* Real    */ ae_vector* bu,
     /* Boolean */ ae_vector* havebu,
     ae_int_t nmain,
     ae_int_t nslack,
     ae_state *_state);
void calculatestepbound(/* Real    */ ae_vector* x,
     /* Real    */ ae_vector* d,
     double alpha,
     /* Real    */ ae_vector* bndl,
     /* Boolean */ ae_vector* havebndl,
     /* Real    */ ae_vector* bndu,
     /* Boolean */ ae_vector* havebndu,
     ae_int_t nmain,
     ae_int_t nslack,
     ae_int_t* variabletofreeze,
     double* valuetofreeze,
     double* maxsteplen,
     ae_state *_state);
ae_int_t postprocessboundedstep(/* Real    */ ae_vector* x,
     /* Real    */ ae_vector* xprev,
     /* Real    */ ae_vector* bndl,
     /* Boolean */ ae_vector* havebndl,
     /* Real    */ ae_vector* bndu,
     /* Boolean */ ae_vector* havebndu,
     ae_int_t nmain,
     ae_int_t nslack,
     ae_int_t variabletofreeze,
     double valuetofreeze,
     double steptaken,
     double maxsteplen,
     ae_state *_state);
void filterdirection(/* Real    */ ae_vector* d,
     /* Real    */ ae_vector* x,
     /* Real    */ ae_vector* bndl,
     /* Boolean */ ae_vector* havebndl,
     /* Real    */ ae_vector* bndu,
     /* Boolean */ ae_vector* havebndu,
     /* Real    */ ae_vector* s,
     ae_int_t nmain,
     ae_int_t nslack,
     double droptol,
     ae_state *_state);
ae_int_t numberofchangedconstraints(/* Real    */ ae_vector* x,
     /* Real    */ ae_vector* xprev,
     /* Real    */ ae_vector* bndl,
     /* Boolean */ ae_vector* havebndl,
     /* Real    */ ae_vector* bndu,
     /* Boolean */ ae_vector* havebndu,
     ae_int_t nmain,
     ae_int_t nslack,
     ae_state *_state);
ae_bool findfeasiblepoint(/* Real    */ ae_vector* x,
     /* Real    */ ae_vector* bndl,
     /* Boolean */ ae_vector* havebndl,
     /* Real    */ ae_vector* bndu,
     /* Boolean */ ae_vector* havebndu,
     ae_int_t nmain,
     ae_int_t nslack,
     /* Real    */ ae_matrix* ce,
     ae_int_t k,
     double epsi,
     ae_int_t* qpits,
     ae_int_t* gpaits,
     ae_state *_state);
ae_bool derivativecheck(double f0,
     double df0,
     double f1,
     double df1,
     double f,
     double df,
     double width,
     ae_state *_state);
void estimateparabolicmodel(double absasum,
     double absasum2,
     double mx,
     double mb,
     double md,
     double d1,
     double d2,
     ae_int_t* d1est,
     ae_int_t* d2est,
     ae_state *_state);
void inexactlbfgspreconditioner(/* Real    */ ae_vector* s,
     ae_int_t n,
     /* Real    */ ae_vector* d,
     /* Real    */ ae_vector* c,
     /* Real    */ ae_matrix* w,
     ae_int_t k,
     precbuflbfgs* buf,
     ae_state *_state);
void preparelowrankpreconditioner(/* Real    */ ae_vector* d,
     /* Real    */ ae_vector* c,
     /* Real    */ ae_matrix* w,
     ae_int_t n,
     ae_int_t k,
     precbuflowrank* buf,
     ae_state *_state);
void applylowrankpreconditioner(/* Real    */ ae_vector* s,
     precbuflowrank* buf,
     ae_state *_state);
void smoothnessmonitorinit(smoothnessmonitor* monitor,
     ae_int_t n,
     ae_int_t k,
     ae_bool checksmoothness,
     ae_state *_state);
void smoothnessmonitorstartlinesearch(smoothnessmonitor* monitor,
     /* Real    */ ae_vector* x,
     /* Real    */ ae_vector* fi,
     /* Real    */ ae_matrix* jac,
     ae_state *_state);
void smoothnessmonitorstartlinesearch1u(smoothnessmonitor* monitor,
     /* Real    */ ae_vector* s,
     /* Real    */ ae_vector* invs,
     /* Real    */ ae_vector* x,
     double f0,
     /* Real    */ ae_vector* j0,
     ae_state *_state);
void smoothnessmonitorenqueuepoint(smoothnessmonitor* monitor,
     /* Real    */ ae_vector* d,
     double stp,
     /* Real    */ ae_vector* x,
     /* Real    */ ae_vector* fi,
     /* Real    */ ae_matrix* jac,
     ae_state *_state);
void smoothnessmonitorenqueuepoint1u(smoothnessmonitor* monitor,
     /* Real    */ ae_vector* s,
     /* Real    */ ae_vector* invs,
     /* Real    */ ae_vector* d,
     double stp,
     /* Real    */ ae_vector* x,
     double f0,
     /* Real    */ ae_vector* j0,
     ae_state *_state);
void smoothnessmonitorfinalizelinesearch(smoothnessmonitor* monitor,
     ae_state *_state);
void smoothnessmonitorexportreport(smoothnessmonitor* monitor,
     optguardreport* rep,
     ae_state *_state);
ae_bool smoothnessmonitorcheckgradientatx0(smoothnessmonitor* monitor,
     /* Real    */ ae_vector* unscaledx0,
     /* Real    */ ae_vector* s,
     /* Real    */ ae_vector* bndl,
     /* Real    */ ae_vector* bndu,
     ae_bool hasboxconstraints,
     double teststep,
     ae_state *_state);
void _precbuflbfgs_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _precbuflbfgs_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _precbuflbfgs_clear(void* _p);
void _precbuflbfgs_destroy(void* _p);
void _precbuflowrank_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _precbuflowrank_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _precbuflowrank_clear(void* _p);
void _precbuflowrank_destroy(void* _p);
void _smoothnessmonitor_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _smoothnessmonitor_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _smoothnessmonitor_clear(void* _p);
void _smoothnessmonitor_destroy(void* _p);
#endif
#if defined(AE_COMPILE_SNNLS) || !defined(AE_PARTIAL_BUILD)
void snnlsinit(ae_int_t nsmax,
     ae_int_t ndmax,
     ae_int_t nrmax,
     snnlssolver* s,
     ae_state *_state);
void snnlssetproblem(snnlssolver* s,
     /* Real    */ ae_matrix* a,
     /* Real    */ ae_vector* b,
     ae_int_t ns,
     ae_int_t nd,
     ae_int_t nr,
     ae_state *_state);
void snnlsdropnnc(snnlssolver* s, ae_int_t idx, ae_state *_state);
void snnlssolve(snnlssolver* s,
     /* Real    */ ae_vector* x,
     ae_state *_state);
void _snnlssolver_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _snnlssolver_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _snnlssolver_clear(void* _p);
void _snnlssolver_destroy(void* _p);
#endif
#if defined(AE_COMPILE_SACTIVESETS) || !defined(AE_PARTIAL_BUILD)
void sasinit(ae_int_t n, sactiveset* s, ae_state *_state);
void sassetscale(sactiveset* state,
     /* Real    */ ae_vector* s,
     ae_state *_state);
void sassetprecdiag(sactiveset* state,
     /* Real    */ ae_vector* d,
     ae_state *_state);
void sassetbc(sactiveset* state,
     /* Real    */ ae_vector* bndl,
     /* Real    */ ae_vector* bndu,
     ae_state *_state);
void sassetlc(sactiveset* state,
     /* Real    */ ae_matrix* c,
     /* Integer */ ae_vector* ct,
     ae_int_t k,
     ae_state *_state);
void sassetlcx(sactiveset* state,
     /* Real    */ ae_matrix* cleic,
     ae_int_t nec,
     ae_int_t nic,
     ae_state *_state);
ae_bool sasstartoptimization(sactiveset* state,
     /* Real    */ ae_vector* x,
     ae_state *_state);
void sasexploredirection(sactiveset* state,
     /* Real    */ ae_vector* d,
     double* stpmax,
     ae_int_t* cidx,
     double* vval,
     ae_state *_state);
ae_int_t sasmoveto(sactiveset* state,
     /* Real    */ ae_vector* xn,
     ae_bool needact,
     ae_int_t cidx,
     double cval,
     ae_state *_state);
void sasimmediateactivation(sactiveset* state,
     ae_int_t cidx,
     double cval,
     ae_state *_state);
void sasconstraineddescent(sactiveset* state,
     /* Real    */ ae_vector* g,
     /* Real    */ ae_vector* d,
     ae_state *_state);
void sasconstraineddescentprec(sactiveset* state,
     /* Real    */ ae_vector* g,
     /* Real    */ ae_vector* d,
     ae_state *_state);
void sasconstraineddirection(sactiveset* state,
     /* Real    */ ae_vector* d,
     ae_state *_state);
void sasconstraineddirectionprec(sactiveset* state,
     /* Real    */ ae_vector* d,
     ae_state *_state);
void sascorrection(sactiveset* state,
     /* Real    */ ae_vector* x,
     double* penalty,
     ae_state *_state);
double sasactivelcpenalty1(sactiveset* state,
     /* Real    */ ae_vector* x,
     ae_state *_state);
double sasscaledconstrainednorm(sactiveset* state,
     /* Real    */ ae_vector* d,
     ae_state *_state);
void sasstopoptimization(sactiveset* state, ae_state *_state);
void sasreactivateconstraints(sactiveset* state,
     /* Real    */ ae_vector* gc,
     ae_state *_state);
void sasreactivateconstraintsprec(sactiveset* state,
     /* Real    */ ae_vector* gc,
     ae_state *_state);
void sasrebuildbasis(sactiveset* state, ae_state *_state);
void sasappendtobasis(sactiveset* state,
     /* Boolean */ ae_vector* newentries,
     ae_state *_state);
void _sactiveset_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _sactiveset_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _sactiveset_clear(void* _p);
void _sactiveset_destroy(void* _p);
#endif
#if defined(AE_COMPILE_QQPSOLVER) || !defined(AE_PARTIAL_BUILD)
void qqploaddefaults(ae_int_t n, qqpsettings* s, ae_state *_state);
void qqpcopysettings(qqpsettings* src, qqpsettings* dst, ae_state *_state);
void qqppreallocategrowdense(qqpbuffers* sstate,
     ae_int_t nexpected,
     ae_int_t ngrowto,
     ae_state *_state);
void qqpoptimize(convexquadraticmodel* cqmac,
     sparsematrix* sparseac,
     /* Real    */ ae_matrix* denseac,
     ae_int_t akind,
     ae_bool isupper,
     /* Real    */ ae_vector* bc,
     /* Real    */ ae_vector* bndlc,
     /* Real    */ ae_vector* bnduc,
     /* Real    */ ae_vector* sc,
     /* Real    */ ae_vector* xoriginc,
     ae_int_t nc,
     qqpsettings* settings,
     qqpbuffers* sstate,
     /* Real    */ ae_vector* xs,
     ae_int_t* terminationtype,
     ae_state *_state);
void _qqpsettings_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _qqpsettings_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _qqpsettings_clear(void* _p);
void _qqpsettings_destroy(void* _p);
void _qqpbuffers_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _qqpbuffers_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _qqpbuffers_clear(void* _p);
void _qqpbuffers_destroy(void* _p);
#endif
#if defined(AE_COMPILE_MINLBFGS) || !defined(AE_PARTIAL_BUILD)
void minlbfgscreate(ae_int_t n,
     ae_int_t m,
     /* Real    */ ae_vector* x,
     minlbfgsstate* state,
     ae_state *_state);
void minlbfgscreatef(ae_int_t n,
     ae_int_t m,
     /* Real    */ ae_vector* x,
     double diffstep,
     minlbfgsstate* state,
     ae_state *_state);
void minlbfgssetcond(minlbfgsstate* state,
     double epsg,
     double epsf,
     double epsx,
     ae_int_t maxits,
     ae_state *_state);
void minlbfgssetxrep(minlbfgsstate* state,
     ae_bool needxrep,
     ae_state *_state);
void minlbfgssetstpmax(minlbfgsstate* state,
     double stpmax,
     ae_state *_state);
void minlbfgssetscale(minlbfgsstate* state,
     /* Real    */ ae_vector* s,
     ae_state *_state);
void minlbfgscreatex(ae_int_t n,
     ae_int_t m,
     /* Real    */ ae_vector* x,
     ae_int_t flags,
     double diffstep,
     minlbfgsstate* state,
     ae_state *_state);
void minlbfgssetprecdefault(minlbfgsstate* state, ae_state *_state);
void minlbfgssetpreccholesky(minlbfgsstate* state,
     /* Real    */ ae_matrix* p,
     ae_bool isupper,
     ae_state *_state);
void minlbfgssetprecdiag(minlbfgsstate* state,
     /* Real    */ ae_vector* d,
     ae_state *_state);
void minlbfgssetprecscale(minlbfgsstate* state, ae_state *_state);
void minlbfgssetprecrankklbfgsfast(minlbfgsstate* state,
     /* Real    */ ae_vector* d,
     /* Real    */ ae_vector* c,
     /* Real    */ ae_matrix* w,
     ae_int_t cnt,
     ae_state *_state);
void minlbfgssetpreclowrankexact(minlbfgsstate* state,
     /* Real    */ ae_vector* d,
     /* Real    */ ae_vector* c,
     /* Real    */ ae_matrix* w,
     ae_int_t cnt,
     ae_state *_state);
ae_bool minlbfgsiteration(minlbfgsstate* state, ae_state *_state);
void minlbfgsoptguardgradient(minlbfgsstate* state,
     double teststep,
     ae_state *_state);
void minlbfgsoptguardsmoothness(minlbfgsstate* state,
     ae_int_t level,
     ae_state *_state);
void minlbfgsoptguardresults(minlbfgsstate* state,
     optguardreport* rep,
     ae_state *_state);
void minlbfgsoptguardnonc1test0results(minlbfgsstate* state,
     optguardnonc1test0report* strrep,
     optguardnonc1test0report* lngrep,
     ae_state *_state);
void minlbfgsoptguardnonc1test1results(minlbfgsstate* state,
     optguardnonc1test1report* strrep,
     optguardnonc1test1report* lngrep,
     ae_state *_state);
void minlbfgsresults(minlbfgsstate* state,
     /* Real    */ ae_vector* x,
     minlbfgsreport* rep,
     ae_state *_state);
void minlbfgsresultsbuf(minlbfgsstate* state,
     /* Real    */ ae_vector* x,
     minlbfgsreport* rep,
     ae_state *_state);
void minlbfgsrestartfrom(minlbfgsstate* state,
     /* Real    */ ae_vector* x,
     ae_state *_state);
void minlbfgsrequesttermination(minlbfgsstate* state, ae_state *_state);
void _minlbfgsstate_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _minlbfgsstate_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _minlbfgsstate_clear(void* _p);
void _minlbfgsstate_destroy(void* _p);
void _minlbfgsreport_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _minlbfgsreport_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _minlbfgsreport_clear(void* _p);
void _minlbfgsreport_destroy(void* _p);
#endif
#if defined(AE_COMPILE_QPDENSEAULSOLVER) || !defined(AE_PARTIAL_BUILD)
void qpdenseaulloaddefaults(ae_int_t nmain,
     qpdenseaulsettings* s,
     ae_state *_state);
void qpdenseauloptimize(convexquadraticmodel* a,
     sparsematrix* sparsea,
     ae_int_t akind,
     ae_bool sparseaupper,
     /* Real    */ ae_vector* b,
     /* Real    */ ae_vector* bndl,
     /* Real    */ ae_vector* bndu,
     /* Real    */ ae_vector* s,
     /* Real    */ ae_vector* xorigin,
     ae_int_t nn,
     /* Real    */ ae_matrix* cleic,
     ae_int_t dnec,
     ae_int_t dnic,
     sparsematrix* scleic,
     ae_int_t snec,
     ae_int_t snic,
     ae_bool renormlc,
     qpdenseaulsettings* settings,
     qpdenseaulbuffers* state,
     /* Real    */ ae_vector* xs,
     ae_int_t* terminationtype,
     ae_state *_state);
void _qpdenseaulsettings_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _qpdenseaulsettings_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _qpdenseaulsettings_clear(void* _p);
void _qpdenseaulsettings_destroy(void* _p);
void _qpdenseaulbuffers_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _qpdenseaulbuffers_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _qpdenseaulbuffers_clear(void* _p);
void _qpdenseaulbuffers_destroy(void* _p);
#endif
#if defined(AE_COMPILE_MINBLEIC) || !defined(AE_PARTIAL_BUILD)
void minbleiccreate(ae_int_t n,
     /* Real    */ ae_vector* x,
     minbleicstate* state,
     ae_state *_state);
void minbleiccreatef(ae_int_t n,
     /* Real    */ ae_vector* x,
     double diffstep,
     minbleicstate* state,
     ae_state *_state);
void minbleicsetbc(minbleicstate* state,
     /* Real    */ ae_vector* bndl,
     /* Real    */ ae_vector* bndu,
     ae_state *_state);
void minbleicsetlc(minbleicstate* state,
     /* Real    */ ae_matrix* c,
     /* Integer */ ae_vector* ct,
     ae_int_t k,
     ae_state *_state);
void minbleicsetcond(minbleicstate* state,
     double epsg,
     double epsf,
     double epsx,
     ae_int_t maxits,
     ae_state *_state);
void minbleicsetscale(minbleicstate* state,
     /* Real    */ ae_vector* s,
     ae_state *_state);
void minbleicsetprecdefault(minbleicstate* state, ae_state *_state);
void minbleicsetprecdiag(minbleicstate* state,
     /* Real    */ ae_vector* d,
     ae_state *_state);
void minbleicsetprecscale(minbleicstate* state, ae_state *_state);
void minbleicsetxrep(minbleicstate* state,
     ae_bool needxrep,
     ae_state *_state);
void minbleicsetdrep(minbleicstate* state,
     ae_bool needdrep,
     ae_state *_state);
void minbleicsetstpmax(minbleicstate* state,
     double stpmax,
     ae_state *_state);
ae_bool minbleiciteration(minbleicstate* state, ae_state *_state);
void minbleicoptguardgradient(minbleicstate* state,
     double teststep,
     ae_state *_state);
void minbleicoptguardsmoothness(minbleicstate* state,
     ae_int_t level,
     ae_state *_state);
void minbleicoptguardresults(minbleicstate* state,
     optguardreport* rep,
     ae_state *_state);
void minbleicoptguardnonc1test0results(minbleicstate* state,
     optguardnonc1test0report* strrep,
     optguardnonc1test0report* lngrep,
     ae_state *_state);
void minbleicoptguardnonc1test1results(minbleicstate* state,
     optguardnonc1test1report* strrep,
     optguardnonc1test1report* lngrep,
     ae_state *_state);
void minbleicresults(minbleicstate* state,
     /* Real    */ ae_vector* x,
     minbleicreport* rep,
     ae_state *_state);
void minbleicresultsbuf(minbleicstate* state,
     /* Real    */ ae_vector* x,
     minbleicreport* rep,
     ae_state *_state);
void minbleicrestartfrom(minbleicstate* state,
     /* Real    */ ae_vector* x,
     ae_state *_state);
void minbleicrequesttermination(minbleicstate* state, ae_state *_state);
void minbleicemergencytermination(minbleicstate* state, ae_state *_state);
void _minbleicstate_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _minbleicstate_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _minbleicstate_clear(void* _p);
void _minbleicstate_destroy(void* _p);
void _minbleicreport_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _minbleicreport_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _minbleicreport_clear(void* _p);
void _minbleicreport_destroy(void* _p);
#endif
#if defined(AE_COMPILE_QPBLEICSOLVER) || !defined(AE_PARTIAL_BUILD)
void qpbleicloaddefaults(ae_int_t nmain,
     qpbleicsettings* s,
     ae_state *_state);
void qpbleiccopysettings(qpbleicsettings* src,
     qpbleicsettings* dst,
     ae_state *_state);
void qpbleicoptimize(convexquadraticmodel* a,
     sparsematrix* sparsea,
     ae_int_t akind,
     ae_bool sparseaupper,
     double absasum,
     double absasum2,
     /* Real    */ ae_vector* b,
     /* Real    */ ae_vector* bndl,
     /* Real    */ ae_vector* bndu,
     /* Real    */ ae_vector* s,
     /* Real    */ ae_vector* xorigin,
     ae_int_t n,
     /* Real    */ ae_matrix* cleic,
     ae_int_t nec,
     ae_int_t nic,
     qpbleicsettings* settings,
     qpbleicbuffers* sstate,
     ae_bool* firstcall,
     /* Real    */ ae_vector* xs,
     ae_int_t* terminationtype,
     ae_state *_state);
void _qpbleicsettings_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _qpbleicsettings_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _qpbleicsettings_clear(void* _p);
void _qpbleicsettings_destroy(void* _p);
void _qpbleicbuffers_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _qpbleicbuffers_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _qpbleicbuffers_clear(void* _p);
void _qpbleicbuffers_destroy(void* _p);
#endif
#if defined(AE_COMPILE_MINQP) || !defined(AE_PARTIAL_BUILD)
void minqpcreate(ae_int_t n, minqpstate* state, ae_state *_state);
void minqpsetlinearterm(minqpstate* state,
     /* Real    */ ae_vector* b,
     ae_state *_state);
void minqpsetquadraticterm(minqpstate* state,
     /* Real    */ ae_matrix* a,
     ae_bool isupper,
     ae_state *_state);
void minqpsetquadratictermsparse(minqpstate* state,
     sparsematrix* a,
     ae_bool isupper,
     ae_state *_state);
void minqpsetstartingpoint(minqpstate* state,
     /* Real    */ ae_vector* x,
     ae_state *_state);
void minqpsetorigin(minqpstate* state,
     /* Real    */ ae_vector* xorigin,
     ae_state *_state);
void minqpsetscale(minqpstate* state,
     /* Real    */ ae_vector* s,
     ae_state *_state);
void minqpsetscaleautodiag(minqpstate* state, ae_state *_state);
void minqpsetalgobleic(minqpstate* state,
     double epsg,
     double epsf,
     double epsx,
     ae_int_t maxits,
     ae_state *_state);
void minqpsetalgodenseaul(minqpstate* state,
     double epsx,
     double rho,
     ae_int_t itscnt,
     ae_state *_state);
void minqpsetalgoquickqp(minqpstate* state,
     double epsg,
     double epsf,
     double epsx,
     ae_int_t maxouterits,
     ae_bool usenewton,
     ae_state *_state);
void minqpsetbc(minqpstate* state,
     /* Real    */ ae_vector* bndl,
     /* Real    */ ae_vector* bndu,
     ae_state *_state);
void minqpsetlc(minqpstate* state,
     /* Real    */ ae_matrix* c,
     /* Integer */ ae_vector* ct,
     ae_int_t k,
     ae_state *_state);
void minqpsetlcsparse(minqpstate* state,
     sparsematrix* c,
     /* Integer */ ae_vector* ct,
     ae_int_t k,
     ae_state *_state);
void minqpsetlcmixed(minqpstate* state,
     /* Real    */ ae_matrix* densec,
     /* Integer */ ae_vector* densect,
     ae_int_t densek,
     sparsematrix* sparsec,
     /* Integer */ ae_vector* sparsect,
     ae_int_t sparsek,
     ae_state *_state);
void minqpoptimize(minqpstate* state, ae_state *_state);
void minqpresults(minqpstate* state,
     /* Real    */ ae_vector* x,
     minqpreport* rep,
     ae_state *_state);
void minqpresultsbuf(minqpstate* state,
     /* Real    */ ae_vector* x,
     minqpreport* rep,
     ae_state *_state);
void minqpsetlineartermfast(minqpstate* state,
     /* Real    */ ae_vector* b,
     ae_state *_state);
void minqpsetquadratictermfast(minqpstate* state,
     /* Real    */ ae_matrix* a,
     ae_bool isupper,
     double s,
     ae_state *_state);
void minqprewritediagonal(minqpstate* state,
     /* Real    */ ae_vector* s,
     ae_state *_state);
void minqpsetstartingpointfast(minqpstate* state,
     /* Real    */ ae_vector* x,
     ae_state *_state);
void minqpsetoriginfast(minqpstate* state,
     /* Real    */ ae_vector* xorigin,
     ae_state *_state);
void _minqpstate_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _minqpstate_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _minqpstate_clear(void* _p);
void _minqpstate_destroy(void* _p);
void _minqpreport_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _minqpreport_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _minqpreport_clear(void* _p);
void _minqpreport_destroy(void* _p);
#endif
#if defined(AE_COMPILE_REVISEDDUALSIMPLEX) || !defined(AE_PARTIAL_BUILD)
void dsssettingsinit(dualsimplexsettings* settings, ae_state *_state);
void dssinit(ae_int_t n, dualsimplexstate* s, ae_state *_state);
void dsssetproblem(dualsimplexstate* state,
     /* Real    */ ae_vector* c,
     /* Real    */ ae_vector* bndl,
     /* Real    */ ae_vector* bndu,
     /* Real    */ ae_vector* sv,
     /* Real    */ ae_matrix* densea,
     sparsematrix* sparsea,
     ae_int_t akind,
     /* Real    */ ae_vector* al,
     /* Real    */ ae_vector* au,
     ae_int_t k,
     dualsimplexbasis* proposedbasis,
     ae_int_t basisinittype,
     dualsimplexsettings* settings,
     ae_state *_state);
void dssexportbasis(dualsimplexstate* state,
     dualsimplexbasis* basis,
     ae_state *_state);
void dssoptimize(dualsimplexstate* state,
     dualsimplexsettings* settings,
     ae_state *_state);
void _dualsimplexsettings_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _dualsimplexsettings_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _dualsimplexsettings_clear(void* _p);
void _dualsimplexsettings_destroy(void* _p);
void _dualsimplexbasis_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _dualsimplexbasis_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _dualsimplexbasis_clear(void* _p);
void _dualsimplexbasis_destroy(void* _p);
void _dualsimplexsubproblem_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _dualsimplexsubproblem_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _dualsimplexsubproblem_clear(void* _p);
void _dualsimplexsubproblem_destroy(void* _p);
void _dualsimplexstate_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _dualsimplexstate_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _dualsimplexstate_clear(void* _p);
void _dualsimplexstate_destroy(void* _p);
#endif
#if defined(AE_COMPILE_MINLP) || !defined(AE_PARTIAL_BUILD)
void minlpcreate(ae_int_t n, minlpstate* state, ae_state *_state);
void minlpsetcost(minlpstate* state,
     /* Real    */ ae_vector* c,
     ae_state *_state);
void minlpsetscale(minlpstate* state,
     /* Real    */ ae_vector* s,
     ae_state *_state);
void minlpsetbc(minlpstate* state,
     /* Real    */ ae_vector* bndl,
     /* Real    */ ae_vector* bndu,
     ae_state *_state);
void minlpsetbcall(minlpstate* state,
     double bndl,
     double bndu,
     ae_state *_state);
void minlpsetbci(minlpstate* state,
     ae_int_t i,
     double bndl,
     double bndu,
     ae_state *_state);
void minlpsetlc(minlpstate* state,
     /* Real    */ ae_matrix* a,
     /* Integer */ ae_vector* ct,
     ae_int_t k,
     ae_state *_state);
void minlpsetlc2dense(minlpstate* state,
     /* Real    */ ae_matrix* a,
     /* Real    */ ae_vector* al,
     /* Real    */ ae_vector* au,
     ae_int_t k,
     ae_state *_state);
void minlpsetlc2(minlpstate* state,
     sparsematrix* a,
     /* Real    */ ae_vector* al,
     /* Real    */ ae_vector* au,
     ae_int_t k,
     ae_state *_state);
void minlpaddlc2dense(minlpstate* state,
     /* Real    */ ae_vector* a,
     double al,
     double au,
     ae_state *_state);
void minlpaddlc2(minlpstate* state,
     /* Integer */ ae_vector* idxa,
     /* Real    */ ae_vector* vala,
     ae_int_t nnz,
     double al,
     double au,
     ae_state *_state);
void minlpoptimize(minlpstate* state, ae_state *_state);
void minlpresults(minlpstate* state,
     /* Real    */ ae_vector* x,
     minlpreport* rep,
     ae_state *_state);
void minlpresultsbuf(minlpstate* state,
     /* Real    */ ae_vector* x,
     minlpreport* rep,
     ae_state *_state);
void _minlpstate_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _minlpstate_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _minlpstate_clear(void* _p);
void _minlpstate_destroy(void* _p);
void _minlpreport_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _minlpreport_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _minlpreport_clear(void* _p);
void _minlpreport_destroy(void* _p);
#endif
#if defined(AE_COMPILE_NLCSLP) || !defined(AE_PARTIAL_BUILD)
void minslpinitbuf(/* Real    */ ae_vector* bndl,
     /* Real    */ ae_vector* bndu,
     /* Real    */ ae_vector* s,
     /* Real    */ ae_vector* x0,
     ae_int_t n,
     /* Real    */ ae_matrix* cleic,
     /* Integer */ ae_vector* lcsrcidx,
     ae_int_t nec,
     ae_int_t nic,
     ae_int_t nlec,
     ae_int_t nlic,
     double epsx,
     ae_int_t maxits,
     minslpstate* state,
     ae_state *_state);
ae_bool minslpiteration(minslpstate* state,
     smoothnessmonitor* smonitor,
     ae_bool userterminationneeded,
     ae_state *_state);
void _minslpsubsolver_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _minslpsubsolver_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _minslpsubsolver_clear(void* _p);
void _minslpsubsolver_destroy(void* _p);
void _minslpstate_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _minslpstate_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _minslpstate_clear(void* _p);
void _minslpstate_destroy(void* _p);
#endif
#if defined(AE_COMPILE_MINNLC) || !defined(AE_PARTIAL_BUILD)
void minnlccreate(ae_int_t n,
     /* Real    */ ae_vector* x,
     minnlcstate* state,
     ae_state *_state);
void minnlccreatef(ae_int_t n,
     /* Real    */ ae_vector* x,
     double diffstep,
     minnlcstate* state,
     ae_state *_state);
void minnlcsetbc(minnlcstate* state,
     /* Real    */ ae_vector* bndl,
     /* Real    */ ae_vector* bndu,
     ae_state *_state);
void minnlcsetlc(minnlcstate* state,
     /* Real    */ ae_matrix* c,
     /* Integer */ ae_vector* ct,
     ae_int_t k,
     ae_state *_state);
void minnlcsetnlc(minnlcstate* state,
     ae_int_t nlec,
     ae_int_t nlic,
     ae_state *_state);
void minnlcsetcond(minnlcstate* state,
     double epsx,
     ae_int_t maxits,
     ae_state *_state);
void minnlcsetscale(minnlcstate* state,
     /* Real    */ ae_vector* s,
     ae_state *_state);
void minnlcsetprecinexact(minnlcstate* state, ae_state *_state);
void minnlcsetprecexactlowrank(minnlcstate* state,
     ae_int_t updatefreq,
     ae_state *_state);
void minnlcsetprecexactrobust(minnlcstate* state,
     ae_int_t updatefreq,
     ae_state *_state);
void minnlcsetprecnone(minnlcstate* state, ae_state *_state);
void minnlcsetstpmax(minnlcstate* state, double stpmax, ae_state *_state);
void minnlcsetalgoaul(minnlcstate* state,
     double rho,
     ae_int_t itscnt,
     ae_state *_state);
void minnlcsetalgoslp(minnlcstate* state, ae_state *_state);
void minnlcsetxrep(minnlcstate* state, ae_bool needxrep, ae_state *_state);
ae_bool minnlciteration(minnlcstate* state, ae_state *_state);
void minnlcoptguardgradient(minnlcstate* state,
     double teststep,
     ae_state *_state);
void minnlcoptguardsmoothness(minnlcstate* state,
     ae_int_t level,
     ae_state *_state);
void minnlcoptguardresults(minnlcstate* state,
     optguardreport* rep,
     ae_state *_state);
void minnlcoptguardnonc1test0results(minnlcstate* state,
     optguardnonc1test0report* strrep,
     optguardnonc1test0report* lngrep,
     ae_state *_state);
void minnlcoptguardnonc1test1results(minnlcstate* state,
     optguardnonc1test1report* strrep,
     optguardnonc1test1report* lngrep,
     ae_state *_state);
void minnlcresults(minnlcstate* state,
     /* Real    */ ae_vector* x,
     minnlcreport* rep,
     ae_state *_state);
void minnlcresultsbuf(minnlcstate* state,
     /* Real    */ ae_vector* x,
     minnlcreport* rep,
     ae_state *_state);
void minnlcrequesttermination(minnlcstate* state, ae_state *_state);
void minnlcrestartfrom(minnlcstate* state,
     /* Real    */ ae_vector* x,
     ae_state *_state);
void minnlcequalitypenaltyfunction(double alpha,
     double* f,
     double* df,
     double* d2f,
     ae_state *_state);
void minnlcinequalitypenaltyfunction(double alpha,
     double stabilizingpoint,
     double* f,
     double* df,
     double* d2f,
     ae_state *_state);
void minnlcinequalityshiftfunction(double alpha,
     double* f,
     double* df,
     double* d2f,
     ae_state *_state);
void _minnlcstate_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _minnlcstate_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _minnlcstate_clear(void* _p);
void _minnlcstate_destroy(void* _p);
void _minnlcreport_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _minnlcreport_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _minnlcreport_clear(void* _p);
void _minnlcreport_destroy(void* _p);
#endif
#if defined(AE_COMPILE_MINBC) || !defined(AE_PARTIAL_BUILD)
void minbccreate(ae_int_t n,
     /* Real    */ ae_vector* x,
     minbcstate* state,
     ae_state *_state);
void minbccreatef(ae_int_t n,
     /* Real    */ ae_vector* x,
     double diffstep,
     minbcstate* state,
     ae_state *_state);
void minbcsetbc(minbcstate* state,
     /* Real    */ ae_vector* bndl,
     /* Real    */ ae_vector* bndu,
     ae_state *_state);
void minbcsetcond(minbcstate* state,
     double epsg,
     double epsf,
     double epsx,
     ae_int_t maxits,
     ae_state *_state);
void minbcsetscale(minbcstate* state,
     /* Real    */ ae_vector* s,
     ae_state *_state);
void minbcsetprecdefault(minbcstate* state, ae_state *_state);
void minbcsetprecdiag(minbcstate* state,
     /* Real    */ ae_vector* d,
     ae_state *_state);
void minbcsetprecscale(minbcstate* state, ae_state *_state);
void minbcsetxrep(minbcstate* state, ae_bool needxrep, ae_state *_state);
void minbcsetstpmax(minbcstate* state, double stpmax, ae_state *_state);
ae_bool minbciteration(minbcstate* state, ae_state *_state);
void minbcoptguardgradient(minbcstate* state,
     double teststep,
     ae_state *_state);
void minbcoptguardsmoothness(minbcstate* state,
     ae_int_t level,
     ae_state *_state);
void minbcoptguardresults(minbcstate* state,
     optguardreport* rep,
     ae_state *_state);
void minbcoptguardnonc1test0results(minbcstate* state,
     optguardnonc1test0report* strrep,
     optguardnonc1test0report* lngrep,
     ae_state *_state);
void minbcoptguardnonc1test1results(minbcstate* state,
     optguardnonc1test1report* strrep,
     optguardnonc1test1report* lngrep,
     ae_state *_state);
void minbcresults(minbcstate* state,
     /* Real    */ ae_vector* x,
     minbcreport* rep,
     ae_state *_state);
void minbcresultsbuf(minbcstate* state,
     /* Real    */ ae_vector* x,
     minbcreport* rep,
     ae_state *_state);
void minbcrestartfrom(minbcstate* state,
     /* Real    */ ae_vector* x,
     ae_state *_state);
void minbcrequesttermination(minbcstate* state, ae_state *_state);
void _minbcstate_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _minbcstate_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _minbcstate_clear(void* _p);
void _minbcstate_destroy(void* _p);
void _minbcreport_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _minbcreport_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _minbcreport_clear(void* _p);
void _minbcreport_destroy(void* _p);
#endif
#if defined(AE_COMPILE_MINNS) || !defined(AE_PARTIAL_BUILD)
void minnscreate(ae_int_t n,
     /* Real    */ ae_vector* x,
     minnsstate* state,
     ae_state *_state);
void minnscreatef(ae_int_t n,
     /* Real    */ ae_vector* x,
     double diffstep,
     minnsstate* state,
     ae_state *_state);
void minnssetbc(minnsstate* state,
     /* Real    */ ae_vector* bndl,
     /* Real    */ ae_vector* bndu,
     ae_state *_state);
void minnssetlc(minnsstate* state,
     /* Real    */ ae_matrix* c,
     /* Integer */ ae_vector* ct,
     ae_int_t k,
     ae_state *_state);
void minnssetnlc(minnsstate* state,
     ae_int_t nlec,
     ae_int_t nlic,
     ae_state *_state);
void minnssetcond(minnsstate* state,
     double epsx,
     ae_int_t maxits,
     ae_state *_state);
void minnssetscale(minnsstate* state,
     /* Real    */ ae_vector* s,
     ae_state *_state);
void minnssetalgoags(minnsstate* state,
     double radius,
     double penalty,
     ae_state *_state);
void minnssetxrep(minnsstate* state, ae_bool needxrep, ae_state *_state);
void minnsrequesttermination(minnsstate* state, ae_state *_state);
ae_bool minnsiteration(minnsstate* state, ae_state *_state);
void minnsresults(minnsstate* state,
     /* Real    */ ae_vector* x,
     minnsreport* rep,
     ae_state *_state);
void minnsresultsbuf(minnsstate* state,
     /* Real    */ ae_vector* x,
     minnsreport* rep,
     ae_state *_state);
void minnsrestartfrom(minnsstate* state,
     /* Real    */ ae_vector* x,
     ae_state *_state);
void _minnsqp_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _minnsqp_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _minnsqp_clear(void* _p);
void _minnsqp_destroy(void* _p);
void _minnsstate_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _minnsstate_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _minnsstate_clear(void* _p);
void _minnsstate_destroy(void* _p);
void _minnsreport_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _minnsreport_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _minnsreport_clear(void* _p);
void _minnsreport_destroy(void* _p);
#endif
#if defined(AE_COMPILE_MINCOMP) || !defined(AE_PARTIAL_BUILD)
void minlbfgssetdefaultpreconditioner(minlbfgsstate* state,
     ae_state *_state);
void minlbfgssetcholeskypreconditioner(minlbfgsstate* state,
     /* Real    */ ae_matrix* p,
     ae_bool isupper,
     ae_state *_state);
void minbleicsetbarrierwidth(minbleicstate* state,
     double mu,
     ae_state *_state);
void minbleicsetbarrierdecay(minbleicstate* state,
     double mudecay,
     ae_state *_state);
void minasacreate(ae_int_t n,
     /* Real    */ ae_vector* x,
     /* Real    */ ae_vector* bndl,
     /* Real    */ ae_vector* bndu,
     minasastate* state,
     ae_state *_state);
void minasasetcond(minasastate* state,
     double epsg,
     double epsf,
     double epsx,
     ae_int_t maxits,
     ae_state *_state);
void minasasetxrep(minasastate* state, ae_bool needxrep, ae_state *_state);
void minasasetalgorithm(minasastate* state,
     ae_int_t algotype,
     ae_state *_state);
void minasasetstpmax(minasastate* state, double stpmax, ae_state *_state);
ae_bool minasaiteration(minasastate* state, ae_state *_state);
void minasaresults(minasastate* state,
     /* Real    */ ae_vector* x,
     minasareport* rep,
     ae_state *_state);
void minasaresultsbuf(minasastate* state,
     /* Real    */ ae_vector* x,
     minasareport* rep,
     ae_state *_state);
void minasarestartfrom(minasastate* state,
     /* Real    */ ae_vector* x,
     /* Real    */ ae_vector* bndl,
     /* Real    */ ae_vector* bndu,
     ae_state *_state);
void _minasastate_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _minasastate_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _minasastate_clear(void* _p);
void _minasastate_destroy(void* _p);
void _minasareport_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _minasareport_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _minasareport_clear(void* _p);
void _minasareport_destroy(void* _p);
#endif
#if defined(AE_COMPILE_MINCG) || !defined(AE_PARTIAL_BUILD)
void mincgcreate(ae_int_t n,
     /* Real    */ ae_vector* x,
     mincgstate* state,
     ae_state *_state);
void mincgcreatef(ae_int_t n,
     /* Real    */ ae_vector* x,
     double diffstep,
     mincgstate* state,
     ae_state *_state);
void mincgsetcond(mincgstate* state,
     double epsg,
     double epsf,
     double epsx,
     ae_int_t maxits,
     ae_state *_state);
void mincgsetscale(mincgstate* state,
     /* Real    */ ae_vector* s,
     ae_state *_state);
void mincgsetxrep(mincgstate* state, ae_bool needxrep, ae_state *_state);
void mincgsetdrep(mincgstate* state, ae_bool needdrep, ae_state *_state);
void mincgsetcgtype(mincgstate* state, ae_int_t cgtype, ae_state *_state);
void mincgsetstpmax(mincgstate* state, double stpmax, ae_state *_state);
void mincgsuggeststep(mincgstate* state, double stp, ae_state *_state);
double mincglastgoodstep(mincgstate* state, ae_state *_state);
void mincgsetprecdefault(mincgstate* state, ae_state *_state);
void mincgsetprecdiag(mincgstate* state,
     /* Real    */ ae_vector* d,
     ae_state *_state);
void mincgsetprecscale(mincgstate* state, ae_state *_state);
ae_bool mincgiteration(mincgstate* state, ae_state *_state);
void mincgoptguardgradient(mincgstate* state,
     double teststep,
     ae_state *_state);
void mincgoptguardsmoothness(mincgstate* state,
     ae_int_t level,
     ae_state *_state);
void mincgoptguardresults(mincgstate* state,
     optguardreport* rep,
     ae_state *_state);
void mincgoptguardnonc1test0results(mincgstate* state,
     optguardnonc1test0report* strrep,
     optguardnonc1test0report* lngrep,
     ae_state *_state);
void mincgoptguardnonc1test1results(mincgstate* state,
     optguardnonc1test1report* strrep,
     optguardnonc1test1report* lngrep,
     ae_state *_state);
void mincgresults(mincgstate* state,
     /* Real    */ ae_vector* x,
     mincgreport* rep,
     ae_state *_state);
void mincgresultsbuf(mincgstate* state,
     /* Real    */ ae_vector* x,
     mincgreport* rep,
     ae_state *_state);
void mincgrestartfrom(mincgstate* state,
     /* Real    */ ae_vector* x,
     ae_state *_state);
void mincgrequesttermination(mincgstate* state, ae_state *_state);
void mincgsetprecdiagfast(mincgstate* state,
     /* Real    */ ae_vector* d,
     ae_state *_state);
void mincgsetpreclowrankfast(mincgstate* state,
     /* Real    */ ae_vector* d1,
     /* Real    */ ae_vector* c,
     /* Real    */ ae_matrix* v,
     ae_int_t vcnt,
     ae_state *_state);
void mincgsetprecvarpart(mincgstate* state,
     /* Real    */ ae_vector* d2,
     ae_state *_state);
void _mincgstate_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _mincgstate_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _mincgstate_clear(void* _p);
void _mincgstate_destroy(void* _p);
void _mincgreport_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _mincgreport_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _mincgreport_clear(void* _p);
void _mincgreport_destroy(void* _p);
#endif
#if defined(AE_COMPILE_MINLM) || !defined(AE_PARTIAL_BUILD)
void minlmcreatevj(ae_int_t n,
     ae_int_t m,
     /* Real    */ ae_vector* x,
     minlmstate* state,
     ae_state *_state);
void minlmcreatev(ae_int_t n,
     ae_int_t m,
     /* Real    */ ae_vector* x,
     double diffstep,
     minlmstate* state,
     ae_state *_state);
void minlmcreatefgh(ae_int_t n,
     /* Real    */ ae_vector* x,
     minlmstate* state,
     ae_state *_state);
void minlmsetcond(minlmstate* state,
     double epsx,
     ae_int_t maxits,
     ae_state *_state);
void minlmsetxrep(minlmstate* state, ae_bool needxrep, ae_state *_state);
void minlmsetstpmax(minlmstate* state, double stpmax, ae_state *_state);
void minlmsetscale(minlmstate* state,
     /* Real    */ ae_vector* s,
     ae_state *_state);
void minlmsetbc(minlmstate* state,
     /* Real    */ ae_vector* bndl,
     /* Real    */ ae_vector* bndu,
     ae_state *_state);
void minlmsetlc(minlmstate* state,
     /* Real    */ ae_matrix* c,
     /* Integer */ ae_vector* ct,
     ae_int_t k,
     ae_state *_state);
void minlmsetacctype(minlmstate* state,
     ae_int_t acctype,
     ae_state *_state);
ae_bool minlmiteration(minlmstate* state, ae_state *_state);
void minlmoptguardgradient(minlmstate* state,
     double teststep,
     ae_state *_state);
void minlmoptguardresults(minlmstate* state,
     optguardreport* rep,
     ae_state *_state);
void minlmresults(minlmstate* state,
     /* Real    */ ae_vector* x,
     minlmreport* rep,
     ae_state *_state);
void minlmresultsbuf(minlmstate* state,
     /* Real    */ ae_vector* x,
     minlmreport* rep,
     ae_state *_state);
void minlmrestartfrom(minlmstate* state,
     /* Real    */ ae_vector* x,
     ae_state *_state);
void minlmrequesttermination(minlmstate* state, ae_state *_state);
void minlmcreatevgj(ae_int_t n,
     ae_int_t m,
     /* Real    */ ae_vector* x,
     minlmstate* state,
     ae_state *_state);
void minlmcreatefgj(ae_int_t n,
     ae_int_t m,
     /* Real    */ ae_vector* x,
     minlmstate* state,
     ae_state *_state);
void minlmcreatefj(ae_int_t n,
     ae_int_t m,
     /* Real    */ ae_vector* x,
     minlmstate* state,
     ae_state *_state);
void _minlmstepfinder_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _minlmstepfinder_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _minlmstepfinder_clear(void* _p);
void _minlmstepfinder_destroy(void* _p);
void _minlmstate_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _minlmstate_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _minlmstate_clear(void* _p);
void _minlmstate_destroy(void* _p);
void _minlmreport_init(void* _p, ae_state *_state, ae_bool make_automatic);
void _minlmreport_init_copy(void* _dst, void* _src, ae_state *_state, ae_bool make_automatic);
void _minlmreport_clear(void* _p);
void _minlmreport_destroy(void* _p);
#endif

}
#endif

