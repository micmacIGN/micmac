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
#include "Digeo.h"

namespace NS_ParamDigeo
{


/****************************************/
/*                                      */
/*           cConvolSpec                */
/*                                      */
/****************************************/

template <class Type> 
cConvolSpec<Type>::cConvolSpec(tBase* aFilter,int aDeb,int aFin,int aNbShit,bool ForGC) :
    mNbShift (aNbShit),
    mDeb      (aDeb),
    mFin      (aFin),
    mForGC    (ForGC)
{
    for (int aK=aDeb; aK<= aFin ; aK++)
    {
       mCoeffs.push_back(aFilter[aK]);
    }

/*
    if (theVec==0)
       theVec = new std::vector<cConvolSpec<Type> *>;
    theVec->push_back(this);
*/
    theVec.push_back(this);
}


template <class Type>
bool cConvolSpec<Type>::Match(tBase *  aDFilter,int aDeb,int aFin,int  aNbShit,bool ForGC)
{
    if (   
           (aNbShit!=mNbShift)
        || (mDeb!=aDeb)
        || (mFin!=aFin)
        || (mForGC!=ForGC)
       )
       return false;

  for (int aK=aDeb; aK<=aFin ; aK++)
     if (ElAbs(mCoeffs[aK-aDeb]-aDFilter[aK]) >1e-4)
        return false;

  return true;
}


template <class Type>
cConvolSpec<Type> * cConvolSpec<Type>::Get(tBase* aFilter,int aDeb,int aFin,int aNbShift,bool ForGC)
{
   for (int aK=0 ; aK<int(theVec.size()) ; aK++)
      if (theVec[aK]->Match(aFilter,aDeb,aFin,aNbShift,ForGC))
         return theVec[aK];

  return 0;
}

template <class Type>
void cConvolSpec<Type>::Convol(Type * Out,Type * In,int aK0,int aK1)
{
   ELISE_ASSERT(false,"cConvolSpec<Type>::Convol");
}


/*
template <> std::vector<cConvolSpec<U_INT1> *> * cConvolSpec<U_INT1>::theVec(0);
template <> std::vector<cConvolSpec<U_INT2> *> * cConvolSpec<U_INT2>::theVec(0);
template <> std::vector<cConvolSpec<REAL4> *> * cConvolSpec<REAL4>::theVec(0);
template <> std::vector<cConvolSpec<INT> *> * cConvolSpec<INT>::theVec(0);
*/

template <> std::vector<cConvolSpec<U_INT1> *>  cConvolSpec<U_INT1>::theVec(0);
template <> std::vector<cConvolSpec<U_INT2> *>  cConvolSpec<U_INT2>::theVec(0);
template <> std::vector<cConvolSpec<REAL4> *>  cConvolSpec<REAL4>::theVec(0);
template <> std::vector<cConvolSpec<INT> *>  cConvolSpec<INT>::theVec(0);



InstantiateClassTplDigeo(cConvolSpec)


/****************************************/
/*                                      */
/*           cTplImInMem                */
/*                                      */
/****************************************/

std::string ToNCC_Str(int aV)
{
   return aV>=0 ? ToString(aV) : ("M"+ToString(-aV));
}

std::string ToNCC_Str(double aV)
{
   return aV>=0 ? ToString(round_ni(aV*1000)) : ("M"+ToString(round_ni(-aV*1000)));
}


template <class Type>
std::string cTplImInMem<Type>::NameClassConvSpec(tBase* aFilter, int aDeb, int aFin)
{

    static int  aCpt=0;

    std::string aRes =  std::string("cConvolSpec_") 
                      + El_CTypeTraits<Type>::Name() 
                      + std::string("_Num") + ToString(aCpt++) 
                       + std::string("_") + ToString(mKInOct) 
                      + std::string("_") + ToString(mOct.NbImOri()) 
                      + std::string("_") + ToString(mNbShift);
/*



    for (int aK= aDeb; aK<=aFin ; aK++)
        aRes = aRes +  "C"+ ToNCC_Str(aFilter[aK]);
*/

     return aRes;
}

static void LineSym(FILE * aFile,int aVal,int aK)
{
   fprintf(aFile,"                              +   %d*(In[%d]+In[%d])\n",aVal,aK,-aK);
}
static void LineSym(FILE * aFile,double aVal,int aK)
{
   fprintf(aFile,"                              +   %lf*(In[%d]+In[%d])\n",aVal,aK,-aK);
}
static void LineStd(FILE * aFile,int aVal,int aK)
{
   fprintf(aFile,"                              +   %d*(In[%d])\n",aVal,aK);
}
static void LineStd(FILE * aFile,double aVal,int aK)
{
   fprintf(aFile,"                              +   %lf*(In[%d])\n",aVal,aK);
}


static void  PutVal(FILE * aFile,int aVal)
{
   fprintf(aFile,"%d",aVal);
}
static void  PutVal(FILE * aFile,double aVal)
{
   fprintf(aFile,"%lf",aVal);
}


template <class Type> 
void cTplImInMem <Type>::MakeClassConvolSpec
     (
         FILE * aFileH,
         FILE * aFileCpp,
         tBase* aFilter,
         int aDeb,
         int aFin,
         int aNbShit
     )
{
    if (!aFileH) 
       return;

    if (cConvolSpec<Type>::Get(aFilter,aDeb,aFin,aNbShit,true))
    {
        return;
    }
    // std::cout << "xxxxxx--- NEW  "  << aFilter[aDeb] << " " << aFilter[0] <<  " " << aFilter[aFin] << "\n";
    new cConvolSpec<Type>(aFilter,aDeb,aFin,aNbShit,true);

    std::string aNClass = NameClassConvSpec(aFilter,aDeb,aFin);
    std::string aNType = El_CTypeTraits<Type>::Name();
    std::string aNTBase = El_CTypeTraits<tBase>::Name();

    fprintf(aFileH,"class %s : public cConvolSpec<%s>\n",aNClass.c_str(),aNType.c_str());
    fprintf(aFileH,"{\n");
    fprintf(aFileH,"   public :\n");
    fprintf(aFileH,"      void Convol(%s * Out,%s * In,int aK0,int aK1)\n",aNType.c_str(),aNType.c_str());
    fprintf(aFileH,"      {\n");
    fprintf(aFileH,"          In+=aK0;\n");
    fprintf(aFileH,"          Out+=aK0;\n");
    fprintf(aFileH,"          for (int aK=aK0; aK<aK1 ; aK++)\n");
    fprintf(aFileH,"          {\n");
    fprintf(aFileH,"               *(Out++) =  (\n");
    if (El_CTypeTraits<Type>::IsIntType())
       fprintf(aFileH,"                                %d\n",(1<<aNbShit)/2);
    else
       fprintf(aFileH,"                                 0\n");
    for (int aK=aDeb ; aK <=aFin ; aK++)
    {
        if ((-aK>=aDeb) && (-aK<=aFin) && (aK) && (aFilter[aK]==aFilter[-aK]))
        {
            if (aK<0)
               LineSym(aFileH,aFilter[aK],aK);
        }
        else
        {
               LineStd(aFileH,aFilter[aK],aK);
        }
    }
    if (El_CTypeTraits<Type>::IsIntType())
       fprintf(aFileH,"                           )>>%d;\n",aNbShit);
    else
       fprintf(aFileH,"                           );\n");
    fprintf(aFileH,"               In++;\n");
    fprintf(aFileH,"          }\n");
    fprintf(aFileH,"      }\n\n");
    fprintf(aFileH,"      %s(%s * aFilter):\n",aNClass.c_str(),aNTBase.c_str());
    fprintf(aFileH,"           cConvolSpec<%s>(aFilter-(%d),%d,%d,%d,false) ",aNType.c_str(),aDeb,aDeb,aFin,aNbShit);
    fprintf(aFileH,"      {\n");
    fprintf(aFileH,"      }\n");
    fprintf(aFileH,"};\n\n");


    fprintf(aFileCpp,"   {\n");
    fprintf(aFileCpp,"      %s theCoeff[%d] ={",aNTBase.c_str(),aFin-aDeb+1);

    for (int aK=aDeb ; aK <=aFin ; aK++)
    {
         if (aK!=aDeb) fprintf(aFileCpp,",");
         PutVal(aFileCpp,aFilter[aK]);
    }
    fprintf(aFileCpp,"};\n");
    fprintf(aFileCpp,"         new %s(theCoeff);\n",aNClass.c_str());
    fprintf(aFileCpp,"   }\n");
/*
    {
       int theCoeff[9] = {1,24,248,990,1570,990,248,24,1};
       new cConvolSpec_U_INT2_Sig_0_5_12(theCoeff);
    }
*/
}


InstantiateClassTplDigeo(cTplImInMem)

/************************************/

// Nb = 4 ;; 1 24 248 990 1570 990 248 24 1 

#if (0)
class cConvolSpec_U_INT2_Sig_0_5_12 : public cConvolSpec<U_INT2>
{
   public :
      void Convol(U_INT2 * Out,U_INT2 * In,int aK0,int aK1)
      {
           In+=aK0;
           Out+=aK0;
           for (int aK=aK0; aK<aK1 ; aK++)
           {
                *(Out++) =  (
                                2048
                              +    1*(In[-4]+In[4])
                              +   24*(In[-3]+In[3])
                              +  248*(In[-2]+In[2])
                              +  990*(In[-1]+In[1])
                              + 1570*(In[0])
                          ) >> 12;

                 In++;
           }
      }


      cConvolSpec_U_INT2_Sig_0_5_12(int * aFilter) :
            cConvolSpec<U_INT2>(aFilter+4,-4,4,12,false)
      {
      }
};


void cAppliDigeo::InitConvolSpec()
{
    static bool theFirst = true;
    if (! theFirst) return;
    theFirst = false;

    {
       int theCoeff[9] = {1,24,248,990,1570,990,248,24,1};
       new cConvolSpec_U_INT2_Sig_0_5_12(theCoeff);
    }
}
#endif

};



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
