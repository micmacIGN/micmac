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
//#include <process.h>

#ifdef __USE_EL_COMMAND__
/*********************************************************/
/*                                                       */
/*                  cElCommand                           */
/*                                                       */
/*********************************************************/

cElCommand::cElCommand( const char *i_command ){ push_back(string(i_command)); }
cElCommand::cElCommand( const string &i_command ){ push_back(i_command); }
#endif

const string temporarySubdirectory = "Tmp-MM-Dir/";

int Round(double aV,double aSup,double aInf)
{
  double aVal = aV / aSup;
  aVal = aVal - floor(aVal);
  return round_ni((aSup*aVal)/aInf);
}

std::string GetUnikId()
{
   double aTSec = ElTimeOfDay();

   return         ToString(mm_getpid())
          + "_" + ToString(Round(aTSec,1e3,1.0))
          + "_" + ToString(Round(aTSec,1,1e-3))
          + "_" + ToString(Round(aTSec,1e-3,1e-6));
}
const std::string & mm_getstrpid();


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
    for (int aK=0 ; aK<5 ; aK++)
    {
       FILE *  aFP = fopen(aFile.c_str(),"w");
       if (aFP)
       {
          fclose(aFP);
          ELISE_fp::RmFile(aFile);
          return true;
       }
    }
    return false;
}

//  les directory par defaut d'ecriture (install de MicMac) ne permettent pas toujours un acces en erciture
// pour les fichiers temporaires
//
//  Modif MPD met en priorite les directories locales suite a demande de Telecom pour clusterisation des commandes
// afin que des process concurent ne s'ecrasent pas
//

std::string Dir2Write(const std::string  DirChantier)
{
    static bool First = true;
    static std::string aRes;
    if (First)
    {
        First = false;

        aRes =  DirChantier + "Tmp-MM-Dir/TestOpenMMmmmm" + mm_getstrpid();
        if (TestFileOpen(aRes))
           return  DirChantier + "Tmp-MM-Dir/";

        aRes = DirChantier +  "TestOpenMMmmmm" + mm_getstrpid();
        if (TestFileOpen(aRes))
           return  DirChantier;

        for (int aK=0 ; aK<MemoArgc; aK++)
        {
            std::string aDir,aName;
            SplitDirAndFile(aDir,aName,MemoArgv[aK]);
            aRes = aDir + "TestOpenMMmmmm" + mm_getstrpid();
            if (TestFileOpen(aRes))
               return aDir;
        }


        aRes = MMDir() + "TestOpenMMmmmm"+mm_getstrpid();
        if (TestFileOpen(aRes))
           return MMDir();


        ELISE_ASSERT(false,"Cannot find any directoruy to write tmp files");
        
    }
   
    return aRes;
}

void cEl_GPAO::DoComInParal(const std::list<std::string> & aL,std::string  FileMk , int   aNbProc ,bool Exe,bool MoinsK)
{
    if (aNbProc<=0)  
       aNbProc = NbProcSys();

   // Modif MPD, certain process plantent apres qq heures en finissant sur 
   // FAIL IN :
   // "/usr/bin/make" all -f "/home/mpd/MMM/culture3d/TestOpenMMmmmmMkStdMM" -j8
   // Suspecte que c'est du a un "ecrasement" entre les Makefile lance par des process concurents;
   // tente un unique Id sur ces makefiles ...


    if (FileMk==""){
       if ( isUsingSeparateDirectories() )
          FileMk = MMTemporaryDirectory() + "MkStdMM" +GetUnikId();
       else
          FileMk = Dir2Write() + "MkStdMM" +GetUnikId();
    }
    else  if (Exe)
    {
       FileMk = FileMk + GetUnikId();
    }

    

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

	std::string aSilent = " -s ", aMultiProcess = string(ELISE_windows ? "-P" : "-j") + ToString(aNbProc);
	std::string aContinueOnError  =  (MoinsK?"-k":"");
	if (Exe)
	{
		// launchMake( FileMk, "all", aNbProc, (MoinsK?"-k":"") );
		launchMake( FileMk, "all", aNbProc, aSilent + aContinueOnError);
		ELISE_fp::RmFile(FileMk);
	}
	else
		cout << g_externalToolHandler.get( "make" ).callName() << " all -f " << FileMk << ' ' << aMultiProcess << endl;
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
       FileMk = ( isUsingSeparateDirectories()?MMTemporaryDirectory():Dir2Write() ) + "MkStdMM" + GetUnikId();
       // FileMk = MMDir() + "MkStdMM" + GetUnikId();


    cEl_GPAO aGPAO;
    string targetPath = ( isUsingSeparateDirectories()?MMTemporaryDirectory():aDir+aBeforeTarget );
    for (int aK=0 ; aK<int(aSet.size())  ; aK++)
    {
        std::string aTarget = targetPath + aSet[aK] + anAfterTarget;
        std::string aCom = aBeforeCom +  aDir+ aSet[aK] + anAfterCom;
        aGPAO.GetOrCreate(aTarget,aCom);
        aGPAO.TaskOfName("all").AddDep(aTarget);
    }

    aGPAO.GenerateMakeFile(FileMk);

	launchMake( FileMk, "all", aNbProc );
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
         MM3dBinFile_quotes("PastDevlop")+" ",
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

	launchMake( aFileMk, "all", aNbProc );
	if (Supr) ELISE_fp::RmFile(aFileMk);
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

#ifdef __USE_EL_COMMAND__
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

#ifdef __USE_EL_COMMAND__
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
	//dump();
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

void cEl_GPAO::dump( std::ostream &io_ostream ) const
{
	map<string,cElTask*>::const_iterator itMap = mDico.begin();
	while ( itMap!=mDico.end() ){
		const cElTask &task = *(itMap->second);
		// print name from map
		io_ostream << "task [" << itMap->first << "] " << &task << endl;
		// print name from task's internal data
		io_ostream << "\tname = [" << task.mName << ']' << endl;
		// print rules
		io_ostream << "\trules" << endl;
		list<string>::const_iterator itRule = task.mBR.begin();
		while ( itRule!=task.mBR.end() )
			io_ostream << "\t\t[" << *itRule++ << ']' << endl;
		// print dependencies
		io_ostream << "\tdependencies" << endl;
		vector<cElTask*>::const_iterator itDep = task.mDeps.begin();
		while ( itDep!=task.mDeps.end() )
			io_ostream << "\t\t[" << (*itDep++)->mName << ']' << endl;
		itMap++;
	}
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

#ifdef __USE_EL_COMMAND__
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
	
    #ifdef __USE_EL_COMMAND__
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
				string rule = *itBR;
				if ( !rule.empty() && *(itBR->rbegin())=='\\' ) rule.append(" ");
				fprintf( aFP, "\t %s\n", rule.c_str() );
			#else
				fprintf(aFP,"\t %s\n",itBR->c_str());
			#endif
		}
	#endif
}


bool launchMake(const string &i_makefile, const string &i_rule, unsigned int i_nbJobs, const string &i_options, bool i_stopCurrentProgramOnFail)
{
	string nbJobsStr(ELISE_windows ? "-P" : "-j");
	if (i_nbJobs != 0) nbJobsStr.append(ToString((int)i_nbJobs));

	#ifdef __TRACE_SYSTEM__
		if (__TRACE_SYSTEM__ >= 2) nbJobsStr.clear(); // no multithreading

		static int iMakefile = 0;
		string makefileCopyName;
		// look for a filename that is not already used
		do{
			if ( iMakefile>999 ) cerr << "WARNING: there is a lot of makefile copies already" << endl;
			stringstream ss;
			ss << "Makefile" << setw(3) << setfill('0') << iMakefile++;
			makefileCopyName = ss.str();
		}
		while ( ELISE_fp::exist_file( makefileCopyName ) );
		cout << "###copying [" << i_makefile << "] to [" << makefileCopyName << "]" << endl;
		ELISE_fp::copy_file( i_makefile, makefileCopyName, true );
	#endif

	std::string aCom = string("\"")+(g_externalToolHandler.get( "make" ).callName())+"\" " + i_rule + " -f \"" + i_makefile + "\" " + nbJobsStr + " " + i_options;
	return System(aCom, !i_stopCurrentProgramOnFail) == EXIT_SUCCESS;
}

#if ELISE_unix
	#include <sys/time.h>
	#include <sys/resource.h>

	size_t getSystemMemory()
	{
		 long nbPages = sysconf(_SC_PHYS_PAGES);
		 ELISE_DEBUG_ERROR(nbPages < 0, "getTotalSystemMemory", "nbPages == " << nbPages);

		 long pageSize = sysconf(_SC_PAGE_SIZE);
		 ELISE_DEBUG_ERROR(pageSize < 0, "getTotalSystemMemory", "pageSize == " << pageSize);

		 return size_t(nbPages) * size_t(pageSize);
	}

	size_t getUsedMemory()
	{
		struct rusage rUsage;
		#ifdef __DEBUG
			int result =
		#endif
		//~ getrusage(RUSAGE_SELF, &rUsage);
		getrusage(RUSAGE_THREAD, &rUsage);
		ELISE_DEBUG_ERROR(result == 1, "getUsedMemory", "rusage returned -1");

		//~ __OUT("ru_maxrss   = " << humanReadable(rUsage.ru_maxrss));   // maximum resident set size
		//~ __OUT("ru_ixrss    = " << humanReadable(rUsage.ru_ixrss));    // integral shared memory size
		//~ __OUT("ru_idrss    = " << humanReadable(rUsage.ru_idrss));    // integral unshared data size
		//~ __OUT("ru_isrss    = " << humanReadable(rUsage.ru_isrss));    // integral unshared stack size
		//~ __OUT("ru_minflt   = " << humanReadable(rUsage.ru_minflt));   // page reclaims (soft page faults)
		//~ __OUT("ru_majflt   = " << humanReadable(rUsage.ru_majflt));   // page faults (hard page faults)
		//~ __OUT("ru_nswap    = " << humanReadable(rUsage.ru_nswap));    // swap
		//~ __OUT("ru_inblock  = " << humanReadable(rUsage.ru_inblock));  // block input operations
		//~ __OUT("ru_oublock  = " << humanReadable(rUsage.ru_oublock));  // block output operations
		//~ __OUT("ru_msgsnd   = " << humanReadable(rUsage.ru_msgsnd));   // IPC messages sent
		//~ __OUT("ru_msgrcv   = " << humanReadable(rUsage.ru_msgrcv));   // IPC messages received
		//~ __OUT("ru_nsignals = " << humanReadable(rUsage.ru_nsignals)); // signals received
		//~ __OUT("ru_nvcsw    = " << humanReadable(rUsage.ru_nvcsw));    // voluntary context switches
		//~ __OUT("ru_nivcsw   = " << humanReadable(rUsage.ru_nivcsw));   // involuntary context switches
		//~ struct timeval ru_utime; /* user CPU time used */
		//~ struct timeval ru_stime; /* system CPU time used */

		return rUsage.ru_ixrss + rUsage.ru_idrss + rUsage.ru_isrss;
	}
#else
	size_t getSystemMemory()
	{
		ELISE_DEBUG_ERROR(true, "getTotalSystemMemory", "not implemented");
		return 0;
	}

	size_t getUsedMemory()
	{
		ELISE_DEBUG_ERROR(true, "getUsedMemory", "not implemented");
		return 0;
	}
#endif

string humanReadable( size_t aSize )
{
	stringstream ss;
	ss << aSize << " = " << (aSize >> 10) << " Ko" << " = " << (aSize >> 20) << " Mo";
	return ss.str();
}


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
