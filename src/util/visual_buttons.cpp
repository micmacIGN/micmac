#include "general/visual_buttons.h"

#if (ELISE_QT_VERSION >= 4)

int cSelectionButton::buttonNumber=0;

cSelectionButton::cSelectionButton(QString text, QWidget *parent)
    : QPushButton(parent),
      unique_id(buttonNumber++)
{
    setText(text);
    connect(this,SIGNAL(clicked()),this,SLOT(onClick()));
}

void cSelectionButton::onClick()
{
    emit my_click(unique_id);
}

cSaisieButton::cSaisieButton(QString text, int id, QWidget *parent)
    : QPushButton(parent),
      _m_line_id(id)
{
    setText(text);
    connect(this,SIGNAL(clicked()),this,SLOT(onClick()));
}

void cSaisieButton::onClick()
{
    emit my_click(_m_line_id);
}
#endif // ELISE_QT_VERSION >= 4



