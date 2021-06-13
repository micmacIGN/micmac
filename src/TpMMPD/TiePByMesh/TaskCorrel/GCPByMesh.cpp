#include "GCPByMesh.h"
#include <stdio.h>

cOneImgMesure::cOneImgMesure()
{}


//  ============================= **************** =============================
//  *                             ReadXMLMesurePts                             *
//  ============================= **************** =============================
void ReadXMLMesurePts(string aPath, vector<cOneImgMesure*> & aVImgMesure)
{
    cSetOfMesureAppuisFlottants aDico = StdGetFromPCP(aPath,SetOfMesureAppuisFlottants);

       std::list<cMesureAppuiFlottant1Im> & aLMAF = aDico.MesureAppuiFlottant1Im();

       std::cout<<"Reading mesure file..."<<std::flush;

       vector<string> aVNameImg;

       for (std::list<cMesureAppuiFlottant1Im>::iterator iT1 = aLMAF.begin() ; iT1 != aLMAF.end() ; iT1++)
       {

           std::list<cOneMesureAF1I> & aMes = iT1->OneMesureAF1I();
           string aNameIm = iT1->NameIm();
           cout<<endl<<" + Img : "<<aNameIm<<" - NbMes : "<<aMes.size()<<endl;

           if (aMes.size() != 0)
           {
               std::vector<string>::iterator it;
               //size_t index;
               bool found = true;
               if (aVNameImg.size() == 0)
               {
                   it = aVNameImg.end();
                   found = false;
               }
               else
               {
                    it = find (aVNameImg.begin(), aVNameImg.end(), aNameIm);
                    if (it != aVNameImg.end())
                    {
                        //recuperer position of it
                        //index = std::distance(aVNameImg.begin(), it);
                        found = false;
                    }
               }

               if (!found)
               {

                   if (it == aVNameImg.end())
                       aVNameImg.push_back(aNameIm);

                   cOneImgMesure * aImgMesure = new cOneImgMesure();
                   aVImgMesure.push_back(aImgMesure);
                   cout<<"NEW GCPPPPPPPPPP ";
                   for (std::list<cOneMesureAF1I>::iterator iT2 = aMes.begin() ; iT2 != aMes.end() ; iT2++)
                   {
                       std::string aNamePt = iT2->NamePt();
                       Pt2dr aPt = iT2->PtIm();
                       cout<<"  + Pts : "<<aNamePt<<" "<<aPt<<endl;

                       aImgMesure->vMesure.push_back(aPt);
                       aImgMesure->vNamePt.push_back(aNamePt);
                   }
               }
           }
       }
       std::cout<<"done!"<<std::endl;
}
