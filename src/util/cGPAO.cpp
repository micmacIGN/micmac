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

#ifdef __C_EL_COMMAND__
/*********************************************************/
/*                                                       */
/*                  cElCommand                           */
/*                                                       */
/*********************************************************/

cElCommand::cElCommand( const char *i_command ){ push_back(string(i_command)); }
cElCommand::cElCommand( const string &i_command ){ push_back(i_command); }
#endif

/*********************************************************/
/*                                                       */
/*                  cEl_GPAO                             */
/*                                                       */
/*********************************************************/

void cEl_GPAO::DoComInSerie(const std::list<std::string> & aL)
{
    for 
    (
        std::list<std::string>::const_iterator itS=aL.begin();
        itS!=aL.end();
        itS++
    )
    {
         System(*itS);
    }
}

bool TestFileOpen(const std::string & aFile)
{
    FILE *  aFP = fopen(aFile.c_str(),"w");
    if (aFP)
    {
       fclose(aFP);
       ELISE_fp::RmFile(aFile);
       return true;
    }
    return false;
}

//  les directory par defaut d'ecriture (install de MicMac) ne permettent pas toujours un acces en erciture
// pour les fichiers temporaires
std::string Dir2Write()
{
    static bool First = true;
    static std::string aRes;
    if (First)
    {
        First = false;
        aRes = MMDir() + "TestOpenMMmmmm";
        if (TestFileOpen(aRes))
           return MMDir();

        aRes = "./TestOpenMMmmmm";
        if (TestFileOpen(aRes))
           return "./";

        for (int aK=0 ; aK<MemoArgc; aK++)
        {
            std::string aDir,aName;
            SplitDirAndFile(aDir,aName,MemoArgv[aK]);
            aRes = aDir + "TestOpenMMmmmm";
            if (TestFileOpen(aRes))
               return aDir;
        }
        ELISE_ASSERT(false,"Cannot find any directoruy to write tmp files");
        
    }
   
    return aRes;
}

void cEl_GPAO::DoComInParal(const std::list<std::string> & aL,std::string  FileMk , int   aNbProc ,bool Exe,bool MoinsK)
{
    if (aNbProc<=0)  
       aNbProc = NbProcSys();

    if (FileMk=="") 
       FileMk = Dir2Write() + "MkStdMM";

    

    cEl_GPAO aGPAO;
    int aK=0;
    for 
    (
        std::list<std::string>::const_iterator itS=aL.begin();
        itS!=aL.end();
        itS++
    )
    {
         std::string aName ="Task_" + ToString(aK);
         cElTask   & aTsk = aGPAO.NewTask(aName,*itS);
         aGPAO.TaskOfName("all").AddDep(aTsk);
         aK++;
    }

    aGPAO.GenerateMakeFile(FileMk);


    std::string aCom = g_externalToolHandler.get( "make" ).callName()+" all -f " + FileMk + " -j" + ToString(aNbProc) + " ";
    if (MoinsK) aCom = aCom + " -k ";
    if (Exe)
    {
        VoidSystem(aCom.c_str());
        ELISE_fp::RmFile(FileMk);
    }
    else
    {
        std::cout << aCom << "\n";
    }
}




void MkFMapCmd
     (
          const std::string & aDir,
          const std::string & aBeforeTarget,
          const std::string & anAfterTarget,
          const std::string & aBeforeCom,
          const std::string & anAfterCom,
          const std::vector<std::string > &aSet ,
          std::string  FileMk = "",  //
          int   aNbProc = 0  // Def = MM Proc
     )
{
    if (aNbProc<=0)  
       aNbProc = NbProcSys();

    if (FileMk=="") 
       FileMk = MMDir() + "MkStdMM";


    cEl_GPAO aGPAO;
    for (int aK=0 ; aK<int(aSet.size())  ; aK++)
    {
        std::string aTarget = aDir + aBeforeTarget + aSet[aK] + anAfterTarget;
        std::string aCom = aBeforeCom +  aDir+ aSet[aK] + anAfterCom;
        aGPAO.GetOrCreate(aTarget,aCom);
        aGPAO.TaskOfName("all").AddDep(aTarget);
    }

    aGPAO.GenerateMakeFile(FileMk);

    std::string aCom = g_externalToolHandler.get( "make" ).callName()+" all -f " + FileMk + " -j" + ToString(aNbProc);
    VoidSystem(aCom.c_str());
}

void MkFMapCmdFileCoul8B
     (
          const std::string & aDir,
          const std::vector<std::string > &aSet 
     )
{
    MkFMapCmd
    (
        aDir,
        "Tmp-MM-Dir/",
        "_Ch3.tif",
         MMBin() + "PastDevlop ",
         " Coul8B=true",
         aSet
    );
}


void cEl_GPAO::ExeParal(std::string aFileMk,int aNbProc,bool Supr)
{
    if (aNbProc<=0)  
       aNbProc = NbProcSys();
    // On prefere laisser la mkfile sur la dir cur : pb de droit si /usr/local/bin/ par ex.
    // aFileMk = MMDir() + aFileMk;
    GenerateMakeFile(aFileMk);

    std::string aCom = string("\"")+g_externalToolHandler.get( "make" ).callName()+"\" all -f \"" + aFileMk + "\" -j" + ToString(aNbProc);
    if (false)
    {
       std::cout << "CCCC = " << aCom << "\n";
       getchar();
    }
    else
    {
       ::System(aCom.c_str());
       if (Supr)
       {
           ELISE_fp::RmFile(aFileMk);
       }
    }
}


/*********************************************************/
/*                                                       */
/*                  cEl_GPAO                             */
/*                                                       */
/*********************************************************/

cEl_GPAO::cEl_GPAO()
{
    NewTask("all","");
}

cEl_GPAO::~cEl_GPAO()
{
}

#ifdef __C_EL_COMMAND__
	cElTask   & cEl_GPAO::NewTask
				(
					 const std::string &aName,
					 const cElCommand & aBuildingRule
				) 
#else
	cElTask   & cEl_GPAO::NewTask
				(
					 const std::string &aName,
					 const std::string & aBuildingRule
				) 
#endif
{
    cElTask * aTask = mDico[aName];

    if (aTask!=0)
    {
        std::cout << "For name " << aName << "\n";
        ELISE_ASSERT
        (
             false,
             "Multiple task creation"
        );
    }
    aTask = new cElTask (aName,*this,aBuildingRule);
    mDico[aName] = aTask;
    return *aTask;
}

cElTask & cEl_GPAO::TaskOfName(const std::string &aName) 
{
    cElTask * aTask = mDico[aName];
    if (aTask==0)
    {
        std::cout << "For name " << aName << "\n";
        ELISE_ASSERT(false,"TaskOfName undefined");
    }

    return *aTask;
}

#ifdef __C_EL_COMMAND__
	cElTask   & cEl_GPAO::GetOrCreate
				(
					 const std::string &aName,
					 const cElCommand & aBuildingRule
				)
#else
	cElTask   & cEl_GPAO::GetOrCreate
				(
					 const std::string &aName,
					 const std::string & aBuildingRule
				)
#endif
{
    cElTask * aTask = mDico[aName];
    return aTask ? *aTask : NewTask(aName,aBuildingRule);
}



void  cEl_GPAO::GenerateMakeFile(const std::string & aNameFile,bool ModeAdditif)  const
{
   // FILE * aFp = ElFopen(aNameFile.c_str(),ModeAdditif ? "a" : "w");
   FILE * aFp = FopenNN(aNameFile.c_str(),ModeAdditif ? "a" : "w","cEl_GPAO::GenerateMakeFile");
   for
   (
       std::map<std::string,cElTask *>::const_iterator iT=mDico.begin();
       iT!=mDico.end();
       iT++
   )
   {
        iT->second->GenerateMakeFile(aFp);
   }
   ElFclose(aFp);
}

void  cEl_GPAO::GenerateMakeFile(const std::string & aNameFile)  const
{
     GenerateMakeFile(aNameFile,false);
}





/*********************************************************/
/*                                                       */
/*                  cElTask                              */
/*                                                       */
/*********************************************************/


void cElTask::AddDep(cElTask & aDep)
{
    mDeps.push_back(&aDep);
}

void cElTask::AddDep(const std::string & aName)
{
    AddDep(mGPAO.TaskOfName(aName));
}

#ifdef __C_EL_COMMAND__
	cElTask::cElTask
	(
				   const std::string & aName,
				   cEl_GPAO & aGPA0,
				   const cElCommand & aBuildingRule
	)  :
#else
	cElTask::cElTask
	(
				   const std::string & aName,
				   cEl_GPAO & aGPA0,
				   const std::string & aBuildingRule
	)  :
#endif
   mGPAO  (aGPA0),
   mName  ( aName)
{
      mBR.push_back(aBuildingRule);
}

   void cElTask::AddBR(const std::string &aBuildingRule)
   {
	   mBR.push_back(aBuildingRule);
   }

void cElTask::GenerateMakeFile(FILE * aFP) const
{
    fprintf(aFP,"%s : ",mName.c_str());
    for (int aK=0;aK<int(mDeps.size());aK++)
       fprintf(aFP,"%s ", mDeps[aK]->mName.c_str());
    fprintf(aFP,"\n");
	
	#ifdef __C_EL_COMMAND__
		list<string>::const_iterator itToken;
		for 
		(
		   std::list<cElCommand>::const_iterator itBR=mBR.begin();
		   itBR!=mBR.end() ;
		   itBR++
		)
		{
			fprintf( aFP,"\t" );
			int iToken = itBR->size()-1;
			itToken=itBR->begin();
			while ( iToken-- )
				fprintf( aFP,"%s ", protect_spaces(*itToken++).c_str() );

			#if (ELISE_windows)
				// avoid a '\' at the end of a line in a makefile
				if ( *(itToken->rbegin())=='\\' )
					//fprintf(aFP,"%s \n", protect_spaces(*itToken).c_str());
					fprintf(aFP,"%s \n", itToken->c_str());
				else
			#endif
			//fprintf(aFP,"%s\n", protect_spaces(*itToken).c_str());
			fprintf(aFP,"%s\n", itToken->c_str());
		}
	#else
		for 
		(
		   std::list<std::string>::const_iterator itBR=mBR.begin();
		   itBR!=mBR.end() ;
		   itBR++
		)
		{
			#if (ELISE_windows)
				// avoid a '\' at the end of a line in a makefile
				if (!itBR->empty())
		
					if ( *(itBR->rbegin())=='\\' )
					{
							string str = *itBR+' ';
							fprintf(aFP,"\t %s\n",str.c_str());
					}
					else
			#endif
			fprintf(aFP,"\t %s\n",itBR->c_str());
		}
	#endif
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
