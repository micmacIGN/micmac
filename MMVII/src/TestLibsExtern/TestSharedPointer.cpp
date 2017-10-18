#include "include/all.h"

namespace NS_TestSharedPointer
{

class cTestMMV2Obj
{
    public :
        virtual ~cTestMMV2Obj()
        {
           TheNbObj--;
        }
        static int NbObj() {return TheNbObj;}
    // protected :
        cTestMMV2Obj()
        {
           TheNbObj++;
        }
    private :
       static int TheNbObj;
};
int cTestMMV2Obj::TheNbObj=0;

void TestCountSharePointer()
{
    {
       std::shared_ptr<cTestMMV2Obj> p0(new cTestMMV2Obj);
       std::shared_ptr<cTestMMV2Obj> p1(new cTestMMV2Obj);

       std::shared_ptr<cTestMMV2Obj> p2 = p0;
       p2 = p1;
       std::cout << "NB OBJ " << cTestMMV2Obj::NbObj() << " =? 2\n";
    }
    std::cout << "NB OBJ " << cTestMMV2Obj::NbObj() << " =? 0\n";
}

    //===============================================

class cFonc1V;

class cDataFonc1V : private cTestMMV2Obj
{
      public :
           virtual ~cDataFonc1V() {}
           virtual double GetVal(double anX) const = 0;
           virtual cFonc1V Derive() const = 0;
           virtual void Show(std::ostream &) const = 0;

           virtual bool IsCste0() const {return false;}
           virtual bool IsCste1() const {return false;}

      public :
};
// std::ostream & operator << (std::ostream & ofs,const Pt2dUi2  &p);


class  cFonc1V : public  std::shared_ptr<cDataFonc1V> 
{
     public :
         double GetVal(double anX)  {return std::shared_ptr<cDataFonc1V>(*this)->GetVal(anX);}
         cFonc1V Derive() const     {return std::shared_ptr<cDataFonc1V>(*this)->Derive();}

         cFonc1V(double aCste);
         cFonc1V operator + (cFonc1V aF2);
         cFonc1V(cDataFonc1V* aPtr) : std::shared_ptr<cDataFonc1V>(aPtr) {}
};
// static cFonc1V X;

//=============================================

class cDataFonc1V_Cste  : public cDataFonc1V
{
      public :
         cDataFonc1V_Cste(double aCste) : mCste(aCste) {}

         double GetVal(double) const {return mCste;}
         cFonc1V Derive() const {return cFonc1V(0.0);}
         void Show(std::ostream & os) const {os << mCste;}

         bool IsCste0() const {return mCste==0;}
         bool IsCste1() const {return mCste==1;}

      private  :
         double mCste;
};
cFonc1V::cFonc1V(double aCste) :
    std::shared_ptr<cDataFonc1V>(new cDataFonc1V_Cste(aCste))
{
}

//=============================================
class cDataFonc1V_X  : public cDataFonc1V
{
      public :
           cDataFonc1V_X() {std::cout << "CREATE X\n";}
           ~cDataFonc1V_X() {std::cout << "Kill X\n";}

           double GetVal(double aVal) const {return aVal;}
           cFonc1V Derive() const {return cFonc1V(1.0);}
           void Show(std::ostream & os) const {os << "X";}
      private  :
};

static cFonc1V X(new cDataFonc1V_X);



//=============================================
class cDataFonc1V_Som  : public cDataFonc1V
{
     public :
         cDataFonc1V_Som(cFonc1V aF1,cFonc1V aF2) : mF1(aF1) , mF2(aF2) {}

         double  GetVal(double aVal) const {return mF1->GetVal(aVal)+mF2->GetVal(aVal);}
         cFonc1V  Derive() const {return mF1->Derive()+mF2->Derive();}
         void Show(std::ostream & os) const 
         {
               os << "(";
               mF1->Show(os); 
               os << "+";
               mF2->Show(os); 
               os << ")";
         }
     public :
         cFonc1V mF1;
         cFonc1V mF2;
};

cFonc1V cFonc1V::operator + (cFonc1V aF2)
{
   return cFonc1V(new cDataFonc1V_Som(*this,aF2));
}



};

using namespace NS_TestSharedPointer;

void  TestSharedPointer()
{
   TestCountSharePointer();

   {
       cFonc1V aF = X+3+(X+4);
       std::cout << "F(10) = " << aF.GetVal(10) << " Count= " << cTestMMV2Obj::NbObj() << "\n";
       std::cout << "F'(10) = " << aF.Derive().GetVal(10)  << "\n";
       aF->Show(std::cout) ; std::cout << "\n";
   }
   std::cout  << " Compte final " << cTestMMV2Obj::NbObj() << "\n";
}

