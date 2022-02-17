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
#include "general/all.h"
#include "private/all.h"
#include <algorithm>


//  bin/Filter $1 "Harris 1.5 2 Moy 1 3" Out=$2


#define DEF_OFSET -12349876

class cOneArgFilter
{
     public :
        cOneArgFilter(const std::string& aName) :
	   mName (aName)
        {
        }

	void AddArg(double  anArg)
	{
             mArgs.push_back(anArg);
	}

	Fonc_Num Calc(Fonc_Num) const;
     private :
	Fonc_Num Itere(Fonc_Num) const;
        void VerifArg(int aMin,int aMax) const;
        std::string mName;
	std::vector<double> mArgs;
};

void cOneArgFilter::VerifArg(int aMin,int aMax) const
{
  if ( (int(mArgs.size()) <aMin) || (int(mArgs.size()) >aMax))
  {
     std::cout << "For " <<  mName << " Nb Arg shoul be in [" <<  aMin << ","<<aMax << "]\n";
     std::cout << " Got " << mArgs.size() << "\n";
     ELISE_ASSERT(false,"Bas Nb Arg to Filtre");
  }
}


Fonc_Num cOneArgFilter::Itere(Fonc_Num aF) const
{
    int aNb = mArgs.back();
    std::cout << "NB ITER " << aNb << "\n";

    cOneArgFilter aClone(*this);
    aClone.mArgs.pop_back();
    for (int aK=0; aK<aNb ; aK++)
        aF = aClone.Calc(aF);

   return aF;
}

Fonc_Num cOneArgFilter::Calc(Fonc_Num aF) const
{
   std::cout <<  mName << "(";
   for (int aK=0 ; aK<int(mArgs.size()) ; aK++)
   {
      if (aK!=0) std::cout << ",";
      std::cout << mArgs[aK] ;
   }
   std::cout <<")\n";

   if (mName=="Harris")
   {
       VerifArg(2,2);

       return Harris(aF,mArgs[0],mArgs[1]);
   }

   if (mName=="Abs")
   {
       VerifArg(0,0);
       return Abs(aF);
   }

   if (mName=="Inv")
   {
       VerifArg(0,0);
       return -aF;
   }
   if (mName=="Lap")
   {
       VerifArg(0,0);
       return Laplacien(aF);
   }




   if (mName=="Mul")
   {
       VerifArg(1,1);
       return aF*mArgs[0];
   }

   if (mName=="Plus")
   {
       VerifArg(1,1);
       return aF+mArgs[0];
   }




   if (mName=="Gamma")
   {
       VerifArg(1,1);
       return 255 * pow(Abs(aF/255.0),mArgs[0]);
   }


   if (mName=="Moy")
   {
       VerifArg(1,2);
       if (mArgs.size()==2)
          return  Itere(aF);
       else
       {
          int aSz = round_ni(mArgs[0]);
          return rect_som(aF,aSz)/ElSquare(1+2.0*aSz);
       }
   }








   ELISE_ASSERT(false,"Unknowm filter");

   return 0;
}



class cAppliFilter
{
    public :
       cAppliFilter(int argc,char ** argv);
       void DoIt(const std::string &,bool ByPat);
       void DoIt();
    private  :

        std::string mNameIn;
        std::string mNameOut;
        std::string mDir;
        std::string mFilters;
	Pt2di       mP0;
	Pt2di       mSz;
        std::list<cOneArgFilter> mLA;


};

void cAppliFilter::DoIt(const std::string & aName,bool ByPat)
{


    std::string aNameOut;
    if (mNameOut !="")
    {
        if (ByPat)
	{
	    ELISE_ASSERT(false,"cAppliFilter::DoIt By Pat non supporte");
	}
	else
	{
            aNameOut =  mNameOut;
	}
    }
    else
    {
        aNameOut = mDir + "Filter_" + aName;
    }

    Tiff_Im aTifIn = Tiff_Im::StdConv(mDir+aName);
    Fonc_Num aF = aTifIn.in_proj();
    for (std::list<cOneArgFilter>::const_iterator itA=mLA.begin(); itA!=mLA.end() ; itA++)
    {
       aF = itA->Calc(aF);
    }

   aF = Max(0,Min(255,aF));
   Pt2di aSzOut = (mSz.x>0) ? mSz : aTifIn.sz();

   Tiff_Im aTifOut
           (
	        aNameOut.c_str(),
		aSzOut,
		GenIm::u_int1,
		Tiff_Im::No_Compr,
		Tiff_Im::BlackIsZero
	   );

   ELISE_COPY
   (
       aTifOut.all_pts(),
       trans(aF,mP0),
       aTifOut.out()
   );
}

void cAppliFilter::DoIt()
{
    DoIt(mNameIn,false);
}



cAppliFilter::cAppliFilter(int argc,char ** argv) :
   mP0 (0,0),
   mSz (-1,-1)
{
    std::string aFullNameIn;
    ElInitArgMain
    (
	argc,argv,
	LArgMain()  << EAM(aFullNameIn)
	            << EAM(mFilters),
	LArgMain()  << EAM(mP0,"P0",true)
                    << EAM(mSz,"Sz",true)
                    << EAM(mNameOut,"Out",true)
    );	
   SplitDirAndFile(mDir,mNameIn,aFullNameIn);


    stringstream  aStr(argv[2]);
    bool cont = true;
    cOneArgFilter  anArg("ttt");
    int aK=0;


    while (cont)
    {
        std::string  aN;
	aStr >> aN;
	cont = (aN !="");
	if (cont)
	{
	  if (isdigit(aN[0] ))
	  {
	      ELISE_ASSERT(aK!=0,"Erreur de syntaxe, digit en premie chiffre");
	      double aD = atof(aN.c_str());
	      anArg.AddArg(aD);
          }
	  else
	  {
	     if (aK!=0)
	        mLA.push_front(anArg);
	     anArg = cOneArgFilter(aN);
	  }
        }

        aK++;
    }
    mLA.push_front(anArg);
}


int main(int argc,char ** argv)
{
    cAppliFilter anAppli(argc,argv);

    anAppli.DoIt();
    return 0;
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
