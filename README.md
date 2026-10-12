# cmach_socket's ncatbot plugins

这里是cmach_socket的 ncatbot plugins合集，目前有：

1. jmcomic 下载bot

2. wol 远程唤醒bot 

## 1. JmPlugins

群聊触发

Feature:

- 输入 `/jm` + 漫画id，下载漫画并返回pdf

- 输入 `/jmd` + 漫画id , 获取漫画详情，以图片形式返回保证安全

## 2. WolPlugins

私聊触发

Feature:

- 输入 `/wol` ，远程唤醒设备

## 3. ClassHelper

班级助手

群聊触发

Feature:

- 将有用的消息转发到班级群

- 自动为消息设置提醒，在指定时间提醒群成员

## 4. ChatBot

聊天bot

群聊触发

Feature:

- 通过工具tool call, 自行选择回复（而不是通过回复率控制）
    
- hindsight 长期记忆系统 

- 基于模拟重退火算法的上下文窗口管理 

## 5. SendRssSummary

从 RSS 源获取内容并总结发送

私聊消息

Feature:

- 定时推送

## 6. CalNotifier

日历提醒器

私聊消息

Feature:

- 通过 CalDAV 协议获取日历事件，并在事件发生前提醒用户
