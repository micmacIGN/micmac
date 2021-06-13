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


#if ELISE_windows && _MSC_VER<_MSC_VER_2013
	// visual c++ <2013
	double round( double aX ){ return aX<0. ? ceil(aX-0.5) : floor(aX+0.5); }
#endif

void cElErrorHandlor::OnError()
{

}

cElErrorHandlor cElErrorHandlor::TheDefElErrorHandlor;
cElErrorHandlor * TheCurElErrorHandlor = & cElErrorHandlor::TheDefElErrorHandlor;

void BasicErrorHandler()
{
     TheCurElErrorHandlor->OnError();
}

//=========================================

int TheIntFuckingReturnValue=1234567;
char * TheCharPtrFuckingReturnValue=0;

bool TheExitOnBrkp  = false;
// bool TheExitOnBrkp  = false;
bool TheExitOnNan   = false;
bool TheExitOnWarn  = false;
bool TheGoonOnWarn  = false;
bool TheMajickFile  = false;
int  TheNbIterProcess = 1;

void throwError(std::string err)
{
    BasicErrorHandler();
    message_copy_where_error();

    // ShowArgs(); A voir comment moduler, mais pour  l'instant ca complique l lecteure des messages ... MPD

    ncout() << err;

    ncout() << "Bye  (press enter)" << endl;

    EliseBRKP();
}

int GetCharOnBrkp()
{
std::cout << "TTTTTTTtttttttttttttttt\n"; getchar();


   BasicErrorHandler();
   if (TheExitOnBrkp)
      return 0;
   return getchar();
}

void EliseBRKP()
{
    BasicErrorHandler();
    if (!TheExitOnBrkp)
       getchar();
}


bool ELISE_DEBUG_USER = true;
bool ELISE_DEBUG_INTERNAL = false;

void Elise_Error_Exit()
{
    BasicErrorHandler();
    message_copy_where_error();
    for (int k=0; k<10; k++) EliseBRKP();
    ElEXIT(1,"");  // Le seul contexte peut venir de message_copy_where_error qui a rempli si necessaire
}

void elise_internal_error(const char * mes,const char * file,int line)
{
    BasicErrorHandler();
    AddMessErrContext
    (
           std::string("elise_internal_error : ") + mes
        +  std::string(" from line ") + ToString(line) + std::string(" of file") + file
    );

    for (int i = 0; i < 3 ; i++)
        ncout() <<  "INTERNAL ERROR IN ELISE !!!\n";
    ncout() << "\n\n The following error :\n";
    ncout() << "    " << mes << "\n";
    ncout() << "occured at line " << line << " of file " << file  << "\n";

    ncout()  << "please send a bug report \n";

    Elise_Error_Exit();
}


void  elise_test_error(const char * mes,const char * file,int line)
{
    BasicErrorHandler();
    ncout() << "KEEP COOL , everything is under control \n";
    ncout() << "        this is a test-fatal error \n";
    ncout() << "The following error : \n";
    ncout() << "    " << mes ;
    ncout() << "was to occure at line " << line << " of file " << file << "\n";
}



void cEliseFatalErrorHandler::SetCurHandler(cEliseFatalErrorHandler * aH)
{
    CurHandler(aH);
}

cEliseFatalErrorHandler * cEliseFatalErrorHandler::CurHandler()
{
    return CurHandler(0);
}


cEliseFatalErrorHandler * cEliseFatalErrorHandler::CurHandler(cEliseFatalErrorHandler * aH)
{
   static cEliseFatalErrorHandler * aRes = 0;
   if ((aH==0) && (aRes==0))
      aRes = new cEliseFatalErrorHandler;
    else
      aRes = aH;
   return aRes;
}

void cEliseFatalErrorHandler::cEFEH_OnErreur(const char * mes,const char * file,int line)
{
    std::string msg =
           "------------------------------------------------------------\n";
    msg += "|   Sorry, the following FATAL ERROR happened               \n";
    msg += "|                                                           \n";
    msg += "|    " + std::string(mes)  +                               "\n";
    msg += "|                                                           \n";
    msg += "------------------------------------------------------------\n";

    std::stringstream sl, sf;
    sl << line;

    #if ELISE_DEPLOY == 0
        sf << file;
    #else
        const char *s = strstr(file, PROJECT_SOURCE_DIR);
        if (s == NULL) sf << file;
        else sf << s + strlen(PROJECT_SOURCE_DIR) + 1;
    #endif

    msg += "-------------------------------------------------------------\n";
    msg += "|       (Elise's)  LOCATION :                                \n";
    msg += "|                                                            \n";
    msg += "| Error was detected\n";
    msg += "|          at line : " + sl.str()  +                        "\n";
    msg += "|          of file : " + sf.str()  +                        "\n";
    msg += "-------------------------------------------------------------\n";
// getchar();

    throwError(msg);

    AddMessErrContext(std::string("mes=") + mes + std::string(" line=") + ToString(line) + std::string(" file=") + file);
    ElEXIT ( 1, "cEliseFatalErrorHandler::cEFEH_OnErreur");
}

void  elise_fatal_error(const char * mes,const char * file,int line)
{
   BasicErrorHandler();
   cEliseFatalErrorHandler *ptrCurHandler = cEliseFatalErrorHandler::CurHandler();
   if (ptrCurHandler != 0)
   {
	   ptrCurHandler->cEFEH_OnErreur(mes, file, line);
   }
}

/*****************************************************************/
/*****************************************************************/
/*****************************************************************/
/*****************************************************************/



std::string ElEM::mes_el() const
{
    std::string mes;

    switch(_type)
    {
        case _int    : mes += ToString(_data.i)     ; break;
        case _real   : mes += ToString(_data.r)     ; break;
        case _string : mes += _data.s     ; break;
        case _pt_pck :  _data.pack->show_kth(_data_2.i);
                        break;
        case _tab_int :
        {
              mes += "[";

              for (INT i  = 0 ; i <_data_2.i; i++)
              {
                  if (i) mes +=  " x ";
                  mes += ToString(_data.Pi[i]);
              }

              mes += "]";
        }
        break;

        case _tab_real :
        {
              mes += "[";

              for (INT i  = 0 ; i <_data_2.i; i++)
              {
                  if (i) mes +=  " x ";
                  mes += ToString(_data.Pr[i]);
              }

              mes += "]";
        }
        break;

        case _pt2di :
        {
            std::stringstream sx, sy;

            sx << _data.pt->x;
            sy << _data.pt->y;

            mes += "(" + sx.str() + "," + sy.str() + ")";
        }
        break;

        case _box_2di :
        {
            std::stringstream p0x, p0y, p1x, p1y;

            const  Box2di * box = _data.box;
            p0x << box->_p0.x;
            p0y << box->_p0.y;
            p1x << box->_p1.x;
            p1y << box->_p1.y;

            mes += "[" + p0x.str() + "," + p1x.str() + "]"
                    + "X"
                    + "[" + p0y.str() + "," + p1y.str() + "]";
        }
        break;

        default:
        ;
    };

    return mes;
}

Elise_Pile_Mess_N::Elise_Pile_Mess_N() {}

Elise_Pile_Mess_N Elise_Pile_Mess_N::_the_one;



void Elise_Pile_Mess_0::display(const char * kind_of)
{ 
    BasicErrorHandler();
    std::string msg =
           "-----------------------------------------------------------------\n";
    msg += "|   KIND OF ERR : " + string(kind_of) +                         "\n";
    msg += "|   Sorry, the following FATAL ERROR happened                    \n";
    msg += "|                                                                \n";
    msg += "|    ";
    for (INT i=0 ; i<_nb ; i++)
    {
         msg += _stack[i].mes_el();
    }
    msg += "\n";
    msg += "|                                                                \n";
    msg += "-----------------------------------------------------------------\n";


    throwError(msg);

    AddMessErrContext(std::string("Kind of err ") + kind_of);
    ElEXIT (EXIT_FAILURE,"Elise_Pile_Mess_0::display");
}

INT Elise_Pile_Mess_0::_nb = 0;
ElEM Elise_Pile_Mess_0::_stack[100];



Elise_Pile_Mess_0 EEM0;


Elise_Assertion::Elise_Assertion(const char * kind_of) :
     _active (true),
     _kind_of (kind_of)
{
}


Elise_Assertion Tjs_El_User("User's error");
Elise_Assertion El_User_Dyn("User's dynamic error");

Elise_Assertion El_Internal("Elise's internal error");


void Elise_Assertion::unactive_user()
{
    ELISE_DEBUG_USER = false;
    El_User_Dyn._active = false;
}



/********************  WARNS  *********************/

cElWarning cElWarning::PlanarityInMasq3d("Planarity in Masq3D ");
cElWarning cElWarning::JacobiInCasa("Singular facets in CASA");
cElWarning cElWarning::BehindCam("Point behind camera after initialisation");
cElWarning cElWarning::FocInxifAndMM("Focal length specified both by xif and NKS-Assoc-STD-FOC");
cElWarning cElWarning::CamInxifAndMM("Camera name specified both by xif and NKS-Assoc-STD-FOC");
cElWarning  cElWarning::GeomIncompAdaptF2Or("Incompatible geometry Cible/Xml, AdaptFoncFileOriMnt");
cElWarning cElWarning::GeomPointTooManyMeasured("Too many measures, excess will be ignored");
cElWarning cElWarning::ToVerifNuage("The point verification in nuage where not coherents");
cElWarning cElWarning::TrueRot("Non rotation matrix has been used as a rotation");
cElWarning cElWarning::ScaleInNuageFromP("Possible scale-problem in cElNuage3DMaille::FromParam");
cElWarning cElWarning::AppuisMultipleDefined("Ground point has several measures in same image");


cElWarning cElWarning::OrhoLocOnlyXCste("For now RedrLocAnam only works with X=Cst Anamorphose");

cElWarning cElWarning::ToleranceSurPoseLibre("Tolerance inutile avec ePoseLibre");
cElWarning cElWarning::OnzeParamSigneIncoh("Point on two sides of cam after space ressection");
cElWarning cElWarning::EigenValueInCholeski("Due to numerical instability, detected negative eigen value in leas square");


int cElWarning::mNbTot = 0;

std::vector<cElWarning *> cElWarning::mWarns;


cElWarning::cElWarning(const std::string & aName) :
   mName    (aName),
   mNbWarns (0)
{
}


void cElWarning::AddWarn
     (
         const std::string &  aMes,
         int                  aLine,
         const std::string &  aFile
     )
{
   if (mNbWarns == 0)
   {
      mWarns.push_back(this);
      mMes = aMes;
      mLine = aLine;
      mFile = aFile;
   }
   mNbWarns ++;
   mNbTot++;
}


void cElWarning::ShowOneWarn(FILE  * aFP)
{
    fprintf(aFP,"%d occurence of warn type [%s]\n",mNbWarns,mName.c_str());
    fprintf(aFP,"First context message : %s\n",mMes.c_str());
    fprintf(aFP,"First detected at line :  %d of File %s\n",mLine,mFile.c_str());
}

void cElWarning::ShowWarns(const std::string & aFile)
{
   if (mNbTot == 0)
      return;

   std::cout << "\n\n";
   std::cout << "*** There were " << mNbTot << " warnings of " << mWarns.size() << " different type \n";
   std::cout << "***  See "  << aFile << " for all warn description \n";
   std::cout << "***  First warn occured \n\n";
   mWarns[0]->ShowOneWarn(stdout);

   FILE * aFP= FopenNN(aFile.c_str(),"w","Warning file");

   for (int aK=0 ; aK<int(mWarns.size()) ; aK++)
   {
        fprintf(aFP,"=================================================================\n");
        mWarns[aK]->ShowOneWarn(aFP);
   }

   ElFclose(aFP);

}


/**************************************************/
/*                                                */
/*             cMajickChek                        */
/*                                                */
/**************************************************/

REAL16 PartieFrac(const REAL16 &aV)
{
   return aV-floor(aV);
}

cMajickChek::cMajickChek() :
   mCheck1(0),
   mCheckInv(0),
   mCheck2(0),
   mGotNan (false),
   mGotInf (false)
{
}

void cMajickChek::AddDouble(const REAL16& aV0)
{

// std::cout << " PF " << PartieFrac(0.3) <<  " " <<  PartieFrac(1.3)  << " " << PartieFrac(-0.7) << "\n";


   REAL16 aV = aV0;

   if (std_isnan(aV))
   {
       mGotNan = true;
       aV = 10.9076461;
   }
   else if (isinf(aV))
   {
       mGotInf = true;
       aV = 90.0011111;
   }
/*  MPD SUPPRIMER DEVRAIT PASSER AVEC  #define isinf(x) (!_finite(x)), inch'allah !!
#if (ELISE_windows)
#else
#endif
*/


   Add1Double(mCheck1,aV+0.1234567);
   Add1Double(mCheck2,aV*aV*1.10987654);
   if (aV)
   {
       Add1Double(mCheckInv,1/aV);
   }
}

void cMajickChek::Add1Double(REAL16 & Target,const REAL16 & aV)
{
   REAL16 aF = PartieFrac(aV);
   Target = PartieFrac(Target + aF + Target*aF); // Target*aV : pour rendre non commut
}

char  hexa(int  aV)
{
  if (aV< 10) return '0' + aV;
  if (aV< 16) return 'A' + (aV-10);
  ELISE_ASSERT(false,"Not hexa");
  return 16;
}

std::string cMajickChek::ShortMajId()
{
   REAL16 aV = mCheck1 - mCheckInv + mCheck2;
   unsigned char * aTabC = (unsigned char *) &aV;
   int aNbOct = 10;  // 6 sont inutilise ?
   for (int aK=0 ; aK<aNbOct ; aK++)
   {
        unsigned  char aC = aTabC[aK];
        sMajAscii[2*aK]   = hexa(aC/16);
        sMajAscii[2*aK+1] = hexa(aC%16);
   }
   sMajAscii[2*aNbOct] = 0;
   return std::string(sMajAscii);
}


std::string cMajickChek::MajId()
{

   std::string aRes =  ShortMajId() + (mGotNan ? "-NAN" : (mGotInf ? "-INF" :"--OK"));

   aRes = aRes + "::" + ToString(double(mCheck1)) +  "::" + ToString(double(mCheckInv)) + "::" + ToString(double(mCheck2));

   return aRes;
}


void  cMajickChek::Add(const Pt3dr & aP)
{
   AddDouble(aP.x);
   AddDouble(aP.y);
   AddDouble(aP.z);
}

void  cMajickChek::Add(const ElRotation3D & aR)
{
    Add(aR.tr());
    AddDouble(aR.teta01());
    AddDouble(aR.teta02());
    AddDouble(aR.teta12());
}

void cMajickChek::Add(const std::string & aS)
{
   int aK=0;
   for (const char * aC= aS.c_str(); *aC;aC++)
   {
      AddDouble(*aC + 1/(0.234+aK));
      aK++;
   }
}

void  cMajickChek::Add(cGenSysSurResol & aSys)
{
    int aNbV = aSys.NbVar();
    for (int aKx=0 ; aKx<aNbV ; aKx++)
    {
        AddDouble(aSys.GetElemLin(aKx));
        for (int aKy=0 ; aKy<aNbV ; aKy++)
        {
            AddDouble(aSys.GetElemQuad(aKx,aKy));
        }
    }
}
void  cMajickChek::Add(cSetEqFormelles & aSetEq)
{
    Add(*(aSetEq.Sys()));
}



/*
*/

// mWarns.push_back(this);




/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
