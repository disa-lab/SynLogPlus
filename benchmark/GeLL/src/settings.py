benchmark_settings = {

    'Linux': {
        'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
        'threshold': 2,
        'regex': [r'(\d+\.){3}\d+', r'\d{2}:\d{2}:\d{2}', r'J([a-z]{2})'],
        'epoch': 1,
    },

    'Proxifier': {
        'log_format': '\[<Time>\] <Program> - <Content>',
        'threshold': 7,
        'regex': [r'<\d+\ssec', r'([\w-]+\.)+[\w-]+(:\d+)?', r'\d{2}:\d{2}(:\d{2})*', r'[KGTM]B'],
        'epoch': 1,
    },

    'Apache': {
        'log_format': '\[<Time>\] \[<Level>\] <Content>',
        'threshold': 3,
        'regex': [r'(\d+\.){3}\d+'],
        'epoch': 1,
    },

    'Zookeeper': {
        'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
        'threshold': 4,
        'regex': [r'(/|)(\d+\.){3}\d+(:\d+)?'],
        'epoch': 1,
    },

    'Mac': {
        'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
        'threshold': 10,
        'regex': [r'([\w-]+\.){2,}[\w-]+'],
        'epoch': 2,
    },

    'HealthApp': {
        'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
        'threshold': 7,
        'regex': [],
        'epoch': 2,
    },

    'Hadoop': {
        'log_format': '<Date> <Time> <Level> \[<Process>\] <Component>: <Content>',
        'threshold': 5,
        'regex': [r'(\d+\.){3}\d+'],
        'epoch': 1,
    },

    'HPC': {
        'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
        'threshold': 4,
        'regex': [r'=\d+'],
        'epoch': 1,
    },

    'OpenSSH': {
        'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
        'threshold': 3,
        'regex': [r'(\d+\.){3}\d+', r'([\w-]+\.){2,}[\w-]+'],
        'epoch': 1,
    },

    'OpenStack': {
        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
        'threshold': 6,
        'regex': [r'((\d+\.){3}\d+,?)+', r'/.+?\s', r'\d+'],
        'epoch': 1,
    },

    'BGL': {
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'threshold': 7,
        'regex': [r'core\.\d+'],
        'epoch': 1,
    },

    'HDFS': {
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'threshold': 5,
        'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
        'epoch': 1,
    },

    'Spark': {
        'log_format': '<Date> <Time> <Level> <Component>: <Content>',
        'threshold': 5,
        'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+'],
        'epoch': 1,
    },

    'Thunderbird': {
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
        'threshold': 4,
        'regex': [r'(\d+\.){3}\d+'],
        'epoch': 1,
    },

    'Android': {
        'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
        'threshold': 15,
        'regex': [r'(/[\w-]+)+', r'([\w-]+\.){2,}[\w-]+', r'\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b'],
        'epoch': 2,
    },

    'Windows': {
        'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
        'threshold': 6,
        'regex': [r'0x.*?\s'],
        'epoch': 1,
    },

}
