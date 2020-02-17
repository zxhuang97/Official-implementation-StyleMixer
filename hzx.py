### DEFAULT SETTINGS #########################
#rank           = 
requirements    = (Machine != "r730a.cs.cityu.edu.hk" && Machine != "r730b.cs.cityu.edu.hk" && Machine != "r730c.cs.cityu.edu.hk" )
priority        = 20
notification    = Never
notify_user     = 
getenv          = True
#environment    = HOME=$ENV(HOME)



# # ## JOB ###########################
# executable      = /usr/bin/python3
# #executable      = /bin/bash
# arguments       = train.py --use_iden --mode swap --fusion Add --i 27 --k 3 --tv_weight 0.0 --identity_weight 0.7 --content_weight 1.0
# input           =
# output          = out20.txt
# error           = $(output)_err.txt
# queue

# # ## JOB ###########################
# executable      = /usr/bin/python3
# #executable      = /bin/bash
# arguments       = train2.py --use_iden --mode swap --fusion Add --i 28 --k 3 --tv_weight 0.0 --identity_weight 1.0 --content_weight 1.2 --use_cx --context_weight 1
# input           =
# output          = out21.txt
# error           = $(output)_err.txt
# queue

# # ## JOB ###########################
# executable      = /usr/bin/python3
# #executable      = /bin/bash
# arguments       = train2.py --use_iden --mode swap --fusion Add --i 29 --k 3 --tv_weight 0.0 --identity_weight 1.0 --content_weight 1.1 --style_weight 3
# input           =
# output          = out22.txt
# error           = $(output)_err.txt
# queue

# # ## JOB ###########################
# executable      = /usr/bin/python3
# #executable      = /bin/bash
# arguments       = train2.py --use_iden --mode swap --fusion Add --i 30 --k 3 --tv_weight 0.0 --identity_weight 1.0 --content_weight 1.2 --use_cx --context_weight 5
# input           =
# output          = out21.txt
# error           = $(output)_err.txt
# queue


# # ## JOB ###########################
# executable      = /usr/bin/python3
# #executable      = /bin/bash
# arguments       = train2.py --use_iden --mode swap --fusion Add --i 31 --k 3 --tv_weight 0.0 --identity_weight 1.0 --content_weight 1.2 --use_cx --context_weight 10 --style_weight 0
# input           =
# output          = out21.txt
# error           = $(output)_err.txt
# queue

# # ## JOB ###########################
# executable      = /usr/bin/python3
# #executable      = /bin/bash
# arguments       = train.py --use_iden --mode swap --fusion Add --i 32 --k 3 --tv_weight 0.0 --identity_weight 1.0 --content_weight 1.5 --use_cx --context_weight 3 --style_weight 3
# input           =
# output          = out21.txt
# error           = $(output)_err.txt
# queue

# # ## JOB ###########################
# executable      = /usr/bin/python3
# #executable      = /bin/bash
# arguments       = train2.py --use_iden --mode swap --fusion Add --i 33 --k 3 --tv_weight 0.0 --identity_weight 0.7 --content_weight 1 --use_cx --context_weight 3 --style_weight 3
# input           =
# output          = out22.txt
# error           = $(output)_err.txt
# queue


# # ## JOB ###########################
# executable      = /usr/bin/python3
# #executable      = /bin/bash
# arguments       = train.py --use_iden --mode swap --fusion Add --i 34 --k 3 --tv_weight 0.0 --identity_weight 1.0 --content_weight 2 --use_cx --context_weight 3 --style_weight 2
# input           =
# output          = out23.txt
# error           = $(output)_err.txt
# queue

# # ## JOB ###########################
# executable      = /usr/bin/python3
# #executable      = /bin/bash
# arguments       = train.py --use_iden --mode swap --fusion Add --i 35 --k 3 --tv_weight 0.0 --identity_weight 1.0 --content_weight 3 --use_cx --context_weight 3 --style_weight 2
# input           =
# output          = out24.txt
# error           = $(output)_err.txt
# queue

# # ## JOB ###########################
# executable      = /usr/bin/python3
# #executable      = /bin/bash
# arguments       = train2.py --use_iden --mode swap --fusion Add --i 36 --k 3 --tv_weight 0.0 --identity_weight 0.7 --content_weight 1 --use_cx --context_weight 3 --style_weight 3
# input           =
# output          = out22.txt
# error           = $(output)_err.txt
# queue

# # ## JOB ###########################
# executable      = /usr/bin/python3
# #executable      = /bin/bash
# arguments       = train.py --use_iden --mode swap --fusion Add --i 37 --k 3 --tv_weight 0.0 --identity_weight 1.0 --content_weight 3 --use_cx --context_weight 3 --style_weight 3
# input           =
# output          = out25.txt
# error           = $(output)_err.txt
# queue

# # ## JOB ###########################
# executable      = /usr/bin/python3
# #executable      = /bin/bash
# arguments       = train.py --use_iden --mode swap --fusion Add --i 38 --k 3 --tv_weight 5.0 --identity_weight 1.0 --content_weight 3.0 --use_cx --context_weight 3 --style_weight 3
# input           =
# output          = out26.txt
# error           = $(output)_err.txt
# queue


# # ## JOB ###########################
# executable      = /usr/bin/python3
# #executable      = /bin/bash
# arguments       = train.py --content_dir /public/zixuhuang3/coco --style_dir /public/zixuhuang3/wikiart --tv_weight 0.0 \
# --identity_weight 1 --content_weight 1 --use_cx --context_weight 3 --style_weight 3
# input           =
# output          = out.txt
# error           = $(output)_err.txt
# queue


# # ## JOB ###########################
# executable      = /usr/bin/python3
# #executable      = /bin/bash
# arguments       = train.py --content_dir /public/zixuhuang3/coco --style_dir /public/zixuhuang3/wikiart --tv_weight 0 \
# --identity_weight 1 --content_weight 3 --cx_weight 3 --style_weight 3
# input           =
# output          = out.txt
# error           = $(output)_err.txt
# queue


# # ## JOB ###########################
# executable      = /usr/bin/python3
# #executable      = /bin/bash
# arguments       = train.py --content_dir /public/zixuhuang3/coco --style_dir /public/zixuhuang3/wikiart --tv_weight 5 \
# --identity_weight 1 --content_weight 3 --cx_weight 3 --style_weight 3
# input           =
# output          = out2.txt
# error           = $(output)_err.txt
# queue

# ## JOB ###########################
executable      = /usr/bin/python3
#executable      = /bin/bash
arguments       = train.py --content_dir /public/zixuhuang3/coco --style_dir /public/zixuhuang3/wikiart --tv_weight 0 \
--identity_weight 1 --content_weight 3 --cx_weight 3 --style_weight 2 --num3
input           =
output          = out.txt
error           = $(output)_err.txt
queue


# ## JOB ###########################
executable      = /usr/bin/python3
#executable      = /bin/bash
arguments       = train.py --content_dir /public/zixuhuang3/coco --style_dir /public/zixuhuang3/wikiart --tv_weight 5 \
--identity_weight 1 --content_weight 3 --cx_weight 3 --style_weight 3 --num4
input           =
output          = out2.txt
error           = $(output)_err.txt
queue

# ## JOB ###########################
executable      = /usr/bin/python3
#executable      = /bin/bash
arguments       = train.py --content_dir /public/zixuhuang3/coco --style_dir /public/zixuhuang3/wikiart --tv_weight 0 \
--identity_weight 1 --content_weight 3 --cx_weight 5 --style_weight 2 --num5
input           =
output          = out3.txt
error           = $(output)_err.txt
queue

