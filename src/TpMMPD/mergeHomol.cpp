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

#include "mergehomol.h"

/**
 * Merge Homol : merge two homol directories 
 *
 * Inputs:
 *  - HomolIn pattern
 *  - HomolOut name
 *
 * Output:
 *  - new homol dir
 *
 * Call example:
 *   mm3d MergeHomol "Homol_[12]" Homol_merged
 *
 * Info: jmmuller
 * 
 * */
//----------------------------------------------------------------------------


vector<string> getSubDirList(string ndir)
{
    if (ndir[ndir.size()-1]=='/')
        ndir.resize(ndir.size()-1);
    vector<string> subStrList;
    cElDirectory aDir(ndir.c_str());
    if (!aDir.IsDirectory()) return subStrList;
    while(const char * subname = aDir.GetNextName())
    {
        if (subname[0] != '.')
        {
            string namecompl = ndir+"/"+subname;
            cElDirectory aSubDir(namecompl.c_str());
            if (aSubDir.IsDirectory())
            {
                subStrList.push_back(subname);
            }
        }
    }
    return subStrList;
}

vector<string> getFilesList(string ndir)
{
    if (ndir[ndir.size()-1]=='/')
        ndir.resize(ndir.size()-1);
    vector<string> subStrList;
    cElDirectory aDir(ndir.c_str());
    if (!aDir.IsDirectory()) return subStrList;
    while(const char * subname = aDir.GetNextName())
    {
        if (subname[0] != '.')
        {
            string namecompl = ndir+"/"+subname;
            cElDirectory aSubDir(namecompl.c_str());
            if (!aSubDir.IsDirectory())
            {
                subStrList.push_back(subname);
            }
        }
    }
    return subStrList;
}


set<string> getFilesSet(string ndir)
{
    if (ndir[ndir.size()-1]=='/')
        ndir.resize(ndir.size()-1);
    set<string> subStrSet;
    cElDirectory aDir(ndir.c_str());
    if (!aDir.IsDirectory()) return subStrSet;
    while(const char * subname = aDir.GetNextName())
    {
        if (subname[0] != '.')
        {
            string namecompl = ndir+"/"+subname;
            cElDirectory aSubDir(namecompl.c_str());
            if (!aSubDir.IsDirectory())
            {
                subStrSet.insert(subname);
            }
        }
    }
    return subStrSet;
}



vector<string> getDirListRegex(string pattern)
{
    cElRegex aRegex(pattern,10000);

    vector<string> aAllDirList = getSubDirList(".");
    vector<string> aSelectedDirList;

    for (unsigned int i=0;i<aAllDirList.size();i++)
    {
        if (aRegex.Match(aAllDirList[i]))
            aSelectedDirList.push_back(aAllDirList[i]);
    }

    return aSelectedDirList;
}


void mergePacks(string packName1,string packName2,string packNameOut)
{
    ElPackHomologue aPackIn1 = ElPackHomologue::FromFile(packName1);
    ElPackHomologue aPackIn2 = ElPackHomologue::FromFile(packName2);
    ElPackHomologue aPackOut;

    //add all points of pack1
    for (ElPackHomologue::const_iterator itP1=aPackIn1.begin(); itP1!=aPackIn1.end() ; ++itP1)
    {
        Pt2dr aPa1 = itP1->P1();
        Pt2dr aPb1 = itP1->P2();
        ElCplePtsHomologues aCple(aPa1,aPb1);
        aPackOut.Cple_Add(aCple);
    }
    //add points of pack2 if not in pack1
    for (ElPackHomologue::const_iterator itP2=aPackIn2.begin(); itP2!=aPackIn2.end() ; ++itP2)
    {
        Pt2dr aPa2 = itP2->P1();
        Pt2dr aPb2 = itP2->P2();

        bool found_a=false;
        bool found_b=false;
        for (ElPackHomologue::const_iterator itP1=aPackIn1.begin(); itP1!=aPackIn1.end() ; ++itP1)
        {
            Pt2dr aPa1 = itP1->P1();
            Pt2dr aPb1 = itP1->P2();
            if ((fabs(aPa1.x-aPa2.x)<0.1)&&(fabs(aPa1.y-aPa2.y)<0.1)) found_a=true;
            if ((fabs(aPb1.x-aPb2.x)<0.1)&&(fabs(aPb1.y-aPb2.y)<0.1)) found_b=true;
            if (found_a||found_b) break;
        }
        if ((!found_a)&&(!found_b))
        {
            ElCplePtsHomologues aCple(aPa2,aPb2);
            aPackOut.Cple_Add(aCple);
        }
        if (found_a!=found_b)
        {
            //incoherence
            aPackOut.Cple_RemoveNearest(aPa2,true);
        }
    }
    aPackOut.StdPutInFile(packNameOut);
}

int mergeHomol_main(int argc,char ** argv)
{
    std::string aHomolInPattern="";//input HomolIn pattern
    std::string aHomolOutDirName="";//output Homol dir

    std::cout<<"\nMergeHomol : merge homol directories\n"<<std::endl;
    bool aPurgeOut(1);
    ElInitArgMain
      (
       argc,argv,
       //mandatory arguments
       LArgMain()  << EAMC(aHomolInPattern, "Homol input pattern name",eSAM_IsPatFile)
                   << EAMC(aHomolOutDirName, "Homol output dir name",eSAM_IsDir),
       //optional arguments
       LArgMain()  << EAM(aPurgeOut,"PurgeOut",true, "Purge Output Homol if it exist prior to merge? Default true.")
      );

    if (MMVisualMode) return EXIT_SUCCESS;

    vector<string> listHomolInDir = getDirListRegex(aHomolInPattern);
    ELISE_ASSERT(listHomolInDir.size()>0,"ERROR: No input Homol dir found!");

    std::cout<<"Homols corresponding to regex:"<<std::endl;
    for (unsigned int i=0;i<listHomolInDir.size();i++)
    {
        std::cout<<" - "<<listHomolInDir[i]<<endl;
    }
    std::cout<<endl;

    bool aOutIsIn(0);
    if(find(listHomolInDir.begin(), listHomolInDir.end(), aHomolOutDirName)!=listHomolInDir.end())  aOutIsIn=1;
    // in case the output Homol database is also one of the input, default= erase. option PurgeOut=0: keep and merge
    if (aPurgeOut) aOutIsIn=0;

    std::string aDirWork,aPatIm;
    SplitDirAndFile(aDirWork,aPatIm,".*");

    for (unsigned int i=0;i<listHomolInDir.size();i++)
    {
        listHomolInDir[i]=aDirWork+listHomolInDir[i];
    }

    aHomolOutDirName=aDirWork+aHomolOutDirName;

    if(ELISE_fp::IsDirectory(aHomolOutDirName) & !aOutIsIn)
    {
        std::cout<<"Warning! "<<aHomolOutDirName<<" already exists!"<<std::endl;
        std::cout<<"Removing "<<aHomolOutDirName<<"..."<<std::endl;
        ELISE_fp::PurgeDirRecursif(aHomolOutDirName);
        std::cout<<aHomolOutDirName<<" removed."<<std::endl;
    }
    ELISE_fp::MkDir(aHomolOutDirName);


    vector<string> aHomolOutDirlist;

    vector<string> aHomolInSubDirlist;
    std::cout<<"Reading sub dir list."<<std::flush;
    for (unsigned int iHomol=0;iHomol<listHomolInDir.size();iHomol++)
    {
        aHomolInSubDirlist=getSubDirList(listHomolInDir[iHomol]);
        std::cout<<"."<<std::flush;
        for (unsigned int i=0;i<aHomolInSubDirlist.size();i++)
        {
            //check if aHomolInSubDirlist[i] is in aHomolOutDirlist
            bool found=false;
            for (unsigned int j=0;j<aHomolOutDirlist.size();j++)
            {
                if (aHomolInSubDirlist[i]==aHomolOutDirlist[j])
                {
                    found=true;
                    break;
                }
            }
            if (!found) aHomolOutDirlist.push_back(aHomolInSubDirlist[i]);
        }
    }
    std::cout<<" done."<<std::endl;

    /*std::cout<<"Merged list:"<<std::endl;
    for (unsigned int i=0;i<aHomolOutDirlist.size();i++)
    {
        std::cout<<" - "<<aHomolOutDirlist[i]<<endl;
    }*/


    long aNumCopiedFiles=0;
    long aNumMergedFiles=0;



    int waitbarupdate=aHomolOutDirlist.size()/100;
    int waitbarsize=0;
    if (aHomolOutDirlist.size()<100)
    {
	    waitbarsize=aHomolOutDirlist.size();
	    waitbarupdate=1;
    }else{
    	waitbarsize=aHomolOutDirlist.size()/waitbarupdate+1;
    }
    std::cout<<aHomolOutDirlist.size()<<" subdirectories to create..."<<std::endl;
    std::cout<<"          ";
    for (int i=0;i<waitbarsize;i++)
	    std::cout<<"_";
    std::cout<<" \n";

    std::cout<<"Merging: |"<<std::flush;
    //for each directory, for each input, copy if does not already exist, or merge
    for (unsigned int aNumDir=0;aNumDir<aHomolOutDirlist.size();aNumDir++)
    {
	if (aNumDir%waitbarupdate==0) cout<<"."<<flush;

        string aDirName=aHomolOutDirlist[aNumDir];
        ELISE_fp::MkDir(aHomolOutDirName+"/"+aDirName);

        for (unsigned int iHomol=0;iHomol<listHomolInDir.size();iHomol++)
        {
            set<string> filesOutSet=getFilesSet(aHomolOutDirName+"/"+aDirName);
            //cout<<"Dir name: "<<aDirName<<" "<<filesOutList.size()<<" files."<<endl;

            string aHomolInName=listHomolInDir[iHomol];
            //cout<<" aHomolInName: "<<aHomolInName<<endl;
            vector<string> filesInList=getFilesList(aHomolInName+"/"+aDirName);
            for (unsigned int i=0;i<filesInList.size();i++)
            {
                //search if file in filesOutSet:
                set<string>::iterator itOut;
                itOut=filesOutSet.find(filesInList[i]);
                if (itOut==filesOutSet.end())
                {
                    aNumCopiedFiles++;
                    //cout<<"  copy "<<aHomolInName+"/"+aDirName+"/"+filesInList[i]<<endl;
                    ELISE_fp::copy_file(aHomolInName+"/"+aDirName+"/"+filesInList[i],aHomolOutDirName+"/"+aDirName+"/"+filesInList[i],true);
                }
                else
                {
                    aNumMergedFiles++;
                    //cout<<"  merge "<<aHomolOutDirName+"/"+aDirName+"/"+filesOutList[indexOut]<<" "<<aHomolInName+"/"+aDirName+"/"+filesInList[i]<<endl;
                    mergePacks(
                               aHomolOutDirName+"/"+aDirName+"/"+filesInList[i],
                               aHomolInName+"/"+aDirName+"/"+filesInList[i],
                               aHomolOutDirName+"/"+aDirName+"/"+filesInList[i] );
                }


            }
        }
    }
    cout<<"| Finished!\n"<<aNumCopiedFiles<<" copied files, "<<aNumMergedFiles<<" merged files."<<endl;

    std::cout<<"Quit"<<std::endl;

    return EXIT_SUCCESS;
}

/* Footer-MicMac-eLiSe-25/06/2007

   Ce logiciel est un programme informatique servant a  la mise en
   correspondances d'images pour la reconstruction du relief.

   Ce logiciel est regi par la licence CeCILL-B soumise au droit francais et
   respectant les principes de diffusion des logiciels libres. Vous pouvez
   utiliser, modifier et/ou redistribuer ce programme sous les conditions
   de la licence CeCILL-B telle que diffusee par le CEA, le CNRS et l'INRIA
   sur le site "http://www.cecill.info".

   En contrepartie de l'accessibilite au code source et des droits de copie,
   de modification et de redistribution accordes par cette licence, il n'est
   offert aux utilisateurs qu'une garantie limitee.  Pour les memes raisons,
   seule une responsabilite restreinte pese sur l'auteur du programme,  le
   titulaire des droits patrimoniaux et les concedants successifs.

   A cet egard  l'attention de l'utilisateur est attiree sur les risques
   associes au chargement, a l'utilisation, a la modification et/ou au
   developpement et a la reproduction du logiciel par l'utilisateur etant
   donne sa specificite de logiciel libre, qui peut le rendre complexe a
   manipuler et qui le reserve donc a des developpeurs et des professionnels
   avertis possedant  des  connaissances  informatiques approfondies.  Les
   utilisateurs sont donc invites a charger  et  tester  l'adequation  du
   logiciel a leurs besoins dans des conditions permettant d'assurer la
   securite de leurs systemes et ou de leurs donnees et, plus generalement,
   a l'utiliser et l'exploiter dans les memes conditions de securite.

   Le fait que vous puissiez acceder a cet en-tete signifie que vous avez
   pris connaissance de la licence CeCILL-B, et que vous en avez accepte les
   termes.
   Footer-MicMac-eLiSe-25/06/2007/*/
