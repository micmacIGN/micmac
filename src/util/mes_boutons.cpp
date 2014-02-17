#if(ELISE_QT5)

#include "general/mes_boutons.h"

#include <iostream>


int mon_bouton_parcours::nbr_boutons_exist=0;

mon_bouton_parcours::mon_bouton_parcours(QWidget * parent)
    : QPushButton(parent), id_unique(nbr_boutons_exist++)
{
    connect(this,SIGNAL(clicked()),this,SLOT(onClick()));
}

mon_bouton_parcours::~mon_bouton_parcours()
{

}

void mon_bouton_parcours::onClick()
{
    std::cout<<"coucou bouton "<<id_unique<<std::endl;
    emit mon_click(id_unique);
}

 #endif // ELISE_QT5
