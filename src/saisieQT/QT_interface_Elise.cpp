#include "QT_interface_Elise.h"

cQT_Interface::cQT_Interface(cAppli_SaisiePts &appli, MainWindow *QTMainWindow):
    m_QTMainWindow(QTMainWindow)
{
    mParam = &appli.Param();
    mAppli = &appli;

    mRefInvis = appli.Param().RefInvis().Val();

}

void cQT_Interface::SetInvisRef(bool aVal)
{
    mRefInvis = aVal;

    //TODO:
    /* for (int aKW=0 ; aKW < (int)mWins.size(); aKW++)
    {
        mWins[aKW]->BCaseVR()->SetVal(aVal);
        mWins[aKW]->Redraw();
        mWins[aKW]->ShowVect();
    }*/
}

cCaseNamePoint *cQT_Interface::GetIndexNamePoint()
{


   /* Video_Win aW = mMenuNamePoint->W();
    aW.raise();

    for (int aK=0 ; aK<int(mVNameCase.size()) ; aK++)
    {
        int aGr = (aK%2) ? 255 : 200 ;
        Pt2di aPCase(0,aK);
        mMenuNamePoint->ColorieCase(aPCase,aW.prgb()(aGr,aGr,aGr),1);
        cCaseNamePoint & aCNP = mVNameCase[aK];
        mMenuNamePoint->StringCase(aPCase,aCNP.mFree ?  aCNP.mName : "***" ,true);
    }

    Clik aClk = aW.clik_in();
    //aW.lower();

    Pt2di aKse = mMenuNamePoint->Pt2Case(Pt2di(aClk._pt));
    cCaseNamePoint * aRes =  &(mVNameCase[aKse.y]);

    if (! aRes->mFree) return 0;

    return aRes;*/

    return 0;
}

std::pair<int, string> cQT_Interface::IdNewPts(cCaseNamePoint *aCNP)
{
   int aCptMax = mAppli->GetCptMax() + 1;

   std::string aName = aCNP->mName;
   if (aCNP->mTCP == eCaseAutoNum)
   {
      std::string nameAuto = mParam->NameAuto().Val();
      aName = nameAuto + ToString(aCptMax);
      aCNP->mName = nameAuto + ToString(aCptMax+1);
   }

   if (aCNP->mTCP == eCaseSaisie)
   {
         //mWEnter->raise();
         //ELISE_COPY(mWEnter->all_pts(),P8COL::yellow,mWEnter->odisc());

         // std::cin >> aName ;
         //aName = mWEnter->GetString(Pt2dr(5,15),mWEnter->pdisc()(P8COL::black),mWEnter->pdisc()(P8COL::yellow));
         //mWEnter->lower();
   }

   //mMenuNamePoint->W().lower();

   // std::cout << "cAppli_SaisiePts::IdNewPts " << aCptMax << " " << aName << "\n";
   //std::pair aRes(
   return std::pair<int,std::string>(aCptMax,aName);

}
