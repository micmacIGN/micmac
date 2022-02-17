/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr

   
    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in 
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte 
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/



#include "StdAfx.h"

#if (ELISE_windows)
	#ifdef INT
		#undef INT
	#endif
	#include "Windows.h"
	#include "winbase.h"
	#include "direct.h"
#endif

/*
class cElRanGen
{
   public :
       REAL cNRrandom3 ();
       void cResetNRrand();
       REAL cNRrandC();
       cElRanGen();

   private :
	int inext,inextp;
        int MSEED;
	long ma[56];
	int iff;
        int idum ;

      float ran3 (int * idum);
};
*/

cElRanGen::cElRanGen() :
	MSEED (161803398),
	iff   (0),
	idum  (-1)
{
}
REAL cElRanGen::cNRrandom3 ()
{
     double r =  ran3(&idum);
     return ElMin(r,0.999999);
}

int  NRrandom3 (int aN)
{
   return ElMax(0,ElMin(aN-1,round_ni(NRrandom3()*aN)));
}


REAL cElRanGen::cNRrandC() 
{
    return 2*(cNRrandom3()-0.5);
}
void cElRanGen::cResetNRrand()
{
   idum = -1;
}


#define MBIG 1000000000
#define MZ 0
#define FAC (1.0/MBIG)

void cElRanGen::InitOfTime(int aNb) //  aNb=1000
{
// std::cout << "cElRanGen::InitOfTime::AAAAAAlllll\n"; getchar();


   double aT = ElTimeOfDay();
   aT = aT -round_down(aT);
   aNb  = round_ni(aT * aNb);
   std::cout << "cElRanGen::InitOfTime, GERM = " << aNb << "\n";
   while (aNb>=0)
   {
        NRrandom3();
        aNb--;
   }
}


float cElRanGen::ran3 (int * idum)
{
	/*
	int inext,inextp;
	long ma[56];
	int iff=0;
	*/
	long mj,mk;
	int i,ii,k;

	if (*idum < 0 || iff == 0) {
		iff=1;
		mj=MSEED-(*idum < 0 ? -*idum : *idum);
		mj %= MBIG;
		ma[55]=mj;
		mk=1;
		for (i=1;i<=54;i++) {
			ii=(21*i) % 55;
			ma[ii]=mk;
			mk=mj-mk;
			if (mk < MZ) mk += MBIG;
			mj=ma[ii];
		}
		for (k=1;k<=4;k++)
			for (i=1;i<=55;i++) {
				ma[i] -= ma[1+(i+30) % 55];
				if (ma[i] < MZ) ma[i] += MBIG;
			}
		inext=0;
		inextp=31;
		*idum=1;
	}
	if (++inext == 56) inext=1;
	if (++inextp == 56) inextp=1;
	mj=ma[inext]-ma[inextp];
	if (mj < MZ) mj += MBIG;
	ma[inext]=mj;

if (0)
{
static int aCpt=0; aCpt++;
static  int aMajic =0;
aMajic  = (aMajic+mj) % 1287;
if (0==(aCpt%100))
{
std::cout << "RRRand " << aCpt << " => " << mj << " " << aMajic << "\n"; // getchar();
}
}
	return (float)(mj*FAC);
}

#undef MBIG
#undef MSEED
#undef MZ
#undef FAC

static cElRanGen aRG;
void NRrandom3InitOfTime()
{
   aRG.InitOfTime();
}

REAL NRrandom3 ()
{
     
     return aRG.cNRrandom3();
}
void ResetNRrand()
{
   aRG.cResetNRrand();
}
REAL NRrandC() {return  aRG.cNRrandC();}


REAL NRrandInterv(double aV0,double aV1) 
{
   return aV0 + (aV1-aV0) * NRrandom3();
}




Fonc_Num gauss_noise_1(INT nb)
{

    REAL nb_pts = ElSquare(2*nb+1);
    REAL moy = 0.5;
    REAL ect = 1 / sqrt(12*nb_pts);

    return rect_som(frandr()-moy,nb)/(nb_pts*ect) ;
}
Fonc_Num unif_noise_1(INT nb)
{
    return erfcc(gauss_noise_1(nb));
}




Fonc_Num gauss_noise_2(INT nb)
{
    REAL nb_pts = ElSquare(2*nb+1);

    return rect_som(gauss_noise_1(nb),nb) /(nb_pts * 0.67);
}
Fonc_Num unif_noise_2(INT nb)
{
    return erfcc(gauss_noise_2(nb));
}




Fonc_Num gauss_noise_3(INT nb)
{
    REAL nb_pts = ElSquare(2*nb+1);

    return rect_som(gauss_noise_2(nb),nb) /(nb_pts * 0.83);
}
Fonc_Num unif_noise_3(INT nb)
{
    return erfcc(gauss_noise_3(nb));
}


Fonc_Num gauss_noise(INT sz,int aK)
{
  if (aK==1) return gauss_noise_1(sz);
  if (aK==2) return gauss_noise_2(sz);
  if (aK==3) return gauss_noise_3(sz);
  return gauss_noise_4(sz);
}

Fonc_Num unif_noise(INT sz,int aK)
{
  if (aK==1) return unif_noise_1(sz);
  if (aK==2) return unif_noise_2(sz);
  if (aK==3) return unif_noise_3(sz);
  return unif_noise_4(sz);
}

Fonc_Num fonc_noise(INT sz,int aK,bool unif)
{
   if (unif)
      return unif_noise(sz,aK);
   return gauss_noise(sz,aK);
}




Fonc_Num gauss_noise_4(INT nb)
{
    REAL nb_pts = ElSquare(2*nb+1);

    return rect_som(gauss_noise_3(nb),nb) /(nb_pts * 0.89);
}
Fonc_Num unif_noise_4(INT nb)
{
    return erfcc(gauss_noise_4(nb));
}


Fonc_Num gauss_noise_4(REAL * pds,INT * sz , INT nb)
{

    Fonc_Num f0 = 0;
    REAL s0 =0;

    for (INT i=0 ; i<nb ; i++)
    {
          f0 =  f0 + pds[i] * gauss_noise_4(sz[i]);
          s0 += ElSquare(pds[i]);
    }

    return f0/sqrt(s0);
}


Fonc_Num unif_noise_4(REAL * pds,INT * sz , INT nb)
{
    return erfcc(gauss_noise_4(pds,sz,nb));
}


cRandNParmiQ:: cRandNParmiQ(int aN,int aQ) :
   mN (aN),
   mQ (aQ)
{
}


bool cRandNParmiQ::GetNext()
{
   ELISE_ASSERT(mQ!=0,"cRandNParmiQ::GetNext");
   bool aRes =(NRrandom3() * mQ) <= mN;
    mQ--;
    if (aRes)
       mN--;

   return aRes;

}


std::vector<int> RandPermut(int aN)
{
    std::vector<Pt2dr> aV;
    for (int aK=0 ; aK<aN ; aK++)
       aV.push_back(Pt2dr(NRrandom3(),aK));

   std::sort(aV.begin(),aV.end());
   std::vector<int> aRes;
   for (int aK=0 ; aK<aN ; aK++)
       aRes.push_back(round_ni(aV[aK].y));

  return aRes;
}


/*
class ElTimer
{
     private :
        REAL _val;

     public :
        ElTimer();
        REAL  val();
};
*/
                
#if (ELISE_unix || ELISE_MacOs)

#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/times.h>
#include <sys/timeb.h>
#include <ctime>                         
                                    

#if (0)
void  ElTimer::set_val()
{
      struct tms buf;
      times(&buf);
      _uval =   buf.tms_utime;
      _sval =   buf.tms_stime;
}
#define TimeUnit CLOCKS_PER_SEC


void  ElTimer::set_val()
{
      struct timeb buf;
      ftime(&buf);
      _uval =   buf.time + buf.millitm * 1e-3;
      _sval =   _uval;
}
#define TimeUnit 1.0
#endif

void  ElTimer::set_val()
{
      _uval = _sval = ElTimeOfDay();
}
#define TimeUnit 1.0




void ElTimer::reinit()
{
      set_val();
     _uval0 = _uval;
     _sval0 = _sval;
}

ElTimer::ElTimer()
{
    reinit();
}

REAL ElTimer::uval()
{
    set_val();
    return (_uval-_uval0) /  TimeUnit;
}
REAL ElTimer::sval()
{
    set_val();
    return (_sval-_sval0) /  TimeUnit;
}
REAL ElTimer::ValAbs()
{
	set_val();
    return (_uval)/TimeUnit;
}

REAL ElTimeOfDay()
{
    struct timeval tv0;
    struct timezone tvz;       
    gettimeofday(&tv0, &tvz); 

    return tv0.tv_sec + (tv0.tv_usec / 1e6);
}

#else 

// Implementation Windows NT
#include <sys/timeb.h>
#include <ctime>


ElTimer::ElTimer()
{
    reinit();
}

void ElTimer::reinit()
{
	 set_val();
     _uval0 = _uval;
     _sval0 = _sval;
}

void  ElTimer::set_val()
{
      _uval =   ElTimeOfDay();
}

REAL ElTimer::uval()
{
	set_val();
    return (_uval-_uval0);
}
REAL ElTimer::ValAbs()
{
	set_val();
    return (_uval);
}

REAL ElTimer::sval() {return 0.0;}


REAL ElTimeOfDay()
{ 
	struct _timeb timebuffer;
	
	_ftime(&timebuffer);

	return timebuffer.time + (timebuffer.millitm / 1e3);
}

#endif

REAL ElTimer::ValAndInit()
{
    REAL aVal = uval();
    reinit();

    return aVal;
} 

double Delay_Brkp=-1;

void SleepProcess(REAL aDelay)
{
   if (aDelay<=0)
      return ;
   int aIDelay = round_ni(aDelay);
#if (!ELISE_windows)
   sleep(aIDelay);
#else
   Sleep(aIDelay);
#endif
   REAL aRDelay = aDelay-aIDelay;
   if (aRDelay >0)
   {
      ElTimer aChrono;
      while (aChrono.uval() < aRDelay)
      {
      }
   }
}






/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA 
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
