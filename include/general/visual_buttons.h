#ifndef VISUAL_BUTTONS_H
#define VISUAL_BUTTONS_H

#include "CMake_defines.h"

#if(ELISE_QT_VERSION >= 4)

#include <QPushButton>

class selectionButton: public QPushButton
{
    Q_OBJECT

public:
        selectionButton(QString text, QWidget *parent=0);

        QAction *Button;

protected:
        int unique_id;
        static int buttonNumber;

public slots:
        void onClick();

signals:
        void my_click(int);
};

#endif //ELISE_QT_VERSION >= 4

#endif /* VISUAL_BUTTONS_H */
