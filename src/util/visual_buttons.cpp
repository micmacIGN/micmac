#if(ELISE_QT5)

#include "general/visual_buttons.h"

#include <iostream>


int selectionButton::buttonNumber=0;

selectionButton::selectionButton(QWidget *parent)
    : QPushButton(parent),
      unique_id(buttonNumber++)
{
    connect(this,SIGNAL(clicked()),this,SLOT(onClick()));
}

selectionButton::~selectionButton()
{

}

void selectionButton::onClick()
{
    //std::cout<<"bouton unique_id: "<<unique_id<<std::endl;
    emit my_click(unique_id);
}

 #endif // ELISE_QT5
