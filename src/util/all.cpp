#include "StdAfx.h"


// TODO : these global definitions should be placed somewhere else
namespace NS_TestOpBuf{
void TestCorrel
     (
           cInterfaceIm2D &         anImOut,
           const cInterfaceIm2D &   anIm1,
           const cInterfaceIm2D &   anIm2,
           const std::complex<int> & aP0,
           const std::complex<int> & aP1,
           int                       aSzV
           
     )
{
    cArgCorrelRapide2Im anArg(anImOut,anIm1,anIm2);

    cTplOpbBufImage<cArgCorrelRapide2Im>  aOPB
                                          (
                                               anArg,
                                               aP0,aP1,
                                               std::complex<int>(-aSzV,-aSzV),
                                               std::complex<int>(aSzV,aSzV)
                                          );

    aOPB.DoIt();
}

void TestSomRapide
     (
           cInterfaceIm2D &         anImOut,
           const cInterfaceIm2D &   anIm1,
           const std::complex<int> & aP0,
           const std::complex<int> & aP1,
           int                       aSzV
           
     )
{
    cArgSommeRapide1Im  anArg(anIm1);

    cTplOpbBufImage<cArgSommeRapide1Im>  aOPB
                                          (
                                               anArg,
                                               aP0,aP1,
                                               std::complex<int>(-aSzV,-aSzV),
                                               std::complex<int>(aSzV,aSzV)
                                          );

    int aNbPts = (1+2*aSzV) * (1+2*aSzV);
    cCumulSomIm *aCum ;
    while ((aCum = aOPB.GetNext()))
    {
          anImOut.SetValue(aOPB.CurPtOut(),aCum->Som()/aNbPts);
    }
}


void TestSomIteree
     (
           cInterfaceIm2D &         anImOut,
           const cInterfaceIm2D &   anIm1,
           const std::complex<int> & aP0,
           const std::complex<int> & aP1,
           int                       aSzV
           
     )
{
    cArgSommeRapideIteree  anArg(anIm1,aP0,aP1,aSzV);

    cTplOpbBufImage<cArgSommeRapideIteree>  aOPB
                                          (
                                               anArg,
                                               aP0,aP1,
                                               std::complex<int>(-aSzV,-aSzV),
                                               std::complex<int>(aSzV,aSzV)
                                          );

    int aNbPts = (1+2*aSzV) * (1+2*aSzV);
    aNbPts = aNbPts * aNbPts;
    cCumulSomIm *aCum ;
    while ((aCum = aOPB.GetNext()))
    {
          anImOut.SetValue(aOPB.CurPtOut(),aCum->Som()/aNbPts);
    }
}
};

extern const char * theNameVar_ParamMICMAC[];


std::string NoInit;
Pt2dr		aNoPt;

Im2DGen AllocImGen(Pt2di aSz,const std::string & aName)
{
    return D2alloc_im2d(type_im(aName),aSz.x,aSz.y);
}

// print cmd and execute ::system (helps with debugging)
int trace_system( const char *cmd )
{
	cout << "###" << current_program_subcommand() << " calls to [" << cmd << ']' << endl;
	int res = ::system( cmd );
#if ( __VERBOSE__>1 )
	if ( res!=0 )
	{
		string str = cmd;
		if (str.find("ElDcraw" )==string::npos)
			cerr << '[' << cmd << "] errorlevel = " << res << endl;
	}
#endif
	return res;
}

#ifdef __TRACE_SYSTEM__
	int (*system_call)( const char* )=trace_system;
#else
	int (*system_call)( const char* )=::system;
#endif

#if (!ELISE_windows)
	FILE * trace_popen( const char *i_cmd, const char *i_access )
	{
		cout << " popen [" << i_cmd << ']' << endl;
		FILE *res = popen( i_cmd, i_access );
	#if ( __VERBOSE__>1 )
		if ( res==NULL )
		{
			string str = i_cmd;
			if (str.find("ElDcraw" )==string::npos)
				cerr << '[' << i_cmd << "] errorlevel = " << errno << endl;
		}
	#endif
		return res;
	}
	
	#ifdef __TRACE_SYSTEM__
		FILE * (*popen_call)( const char *, const char * )=trace_popen;
	#else
		FILE * (*popen_call)( const char *, const char * )=popen;
	#endif
#endif

