cfg = dict(
    data_root = 'C:/Data/UHD_EEG/',

    subjects = ['S1', 'S2', 'S3', 'S4', 'S5'],
    dominant_hand = ['left','right','right','right','right'],

    mapping = {0: "No instruction", 1: "Rest", 2: "thumb", 3: "index", 4: "middle", 5: "ring", 6: "little"},

    not_ROI_channels = ['c255', 'c256', 'c254', 'c251', 'c239', 'c240', 'c238', 'c235', 'c224', 'c222', 'c223', 'c219', 'c220', 'c221', 'c215', 'c216', 'c217', 'c213', 'c212', 'c211', 'c210', 'c209', 'c112', 'c110', 'c107', 'c108', 'c103', 'c104', 'c105', 'c101', 'c100', 'c99', 'c98', 'c97', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c14', 'c15', 'c16', 'c23', 'c29', 'c26', 'c17', 'c18', 'c20', 'c19', 'c21', 'c24', 'c22', 'c25', 'c28', 'c33', 'c35', 'c38', 'c42', 'c81', 'c34', 'c37', 'c41', 'c45', 'c36', 'c40', 'c44', 'c39', 'c43', 'c145', 'c147', 'c150', 'c154', 'c157', 'c153', 'c149', 'c146', 'c93', 'c159', 'c156', 'c152', 'c148', 'c95', 'c160', 'c158', 'c155', 'c151', 'c96', 'c202', 'c198', 'c195', 'c193'],

    bad_channels = {
        'S1': ['c69', 'c122', 'c170', 'c173', 'c189'],
        'S2': ['c1','c61', 'c96', 'c160','c44'],
        'S3': ['c16', 'c18', 'c42', 'c48', 'c77', 'c82', 'c100', 'c102', 'c104', 'c122', 'c189', 'c202', 'c205', 'c223', 'c240', 'c243', 'c243', 'c248', 'c249', 'c251',  'c252', 'c254', 'c256'],
        'S4': ['c8', 'c15', 'c29', 'c32', 'c61', 'c71', 'c77', 'c90', 'c93', 'c100', 'c102', 'c106','c122', 'c135', 'c138', 'c145', 'c157', 'c183', 'c186', 'c190', 'c202', 'c205', 'c221', 'c221'],
        'S5': ['c151', 'c155', 'c158', 'c198', 'c202']
    },
    # 158 channels
    not_ROI_channel_names = ['c13',  'c27',  'c30',  'c31',  'c32',  'c46',  'c47',  'c48',  'c49',  'c50',  'c51',  'c52',  'c53',  'c54',  'c55',  'c56',  'c57',  'c58',  'c59',  'c60',  'c61',  'c62',  'c63',  'c64',  'c65',  'c66',  'c67',  'c68',  'c69',  'c70',  'c71',  'c72',  'c73',  'c74',  'c75',  'c76',  'c77',  'c78',  'c79',  'c80',  'c82',  'c83',  'c84',  'c85',  'c86',  'c87',  'c88',  'c89',  'c90',  'c91',  'c92',  'c94',  'c102',  'c106',  'c109',  'c111',  'c113',  'c114',  'c115',  'c116',  'c117',  'c118',  'c119',  'c120',  'c121',  'c122',  'c123',  'c124',  'c125',  'c126',  'c127',  'c128',  'c129',  'c130',  'c131',  'c132',  'c133',  'c134',  'c135',  'c136',  'c137',  'c138',  'c139',  'c140',  'c141',  'c142',  'c143',  'c144',  'c161',  'c162',  'c163',  'c164',  'c165',  'c166',  'c167',  'c168',  'c169',  'c170',  'c171',  'c172',  'c173',  'c174',  'c175',  'c176',  'c177',  'c178',  'c179',  'c180',  'c181',  'c182',  'c183',  'c184',  'c185',  'c186',  'c187',  'c188',  'c189',  'c190',  'c191',  'c192',  'c194',  'c196',  'c197',  'c199',  'c200',  'c201',  'c203',  'c204',  'c205',  'c206',  'c207',  'c208',  'c214',  'c218',  'c225',  'c226',  'c227',  'c228',  'c229',  'c230',  'c231',  'c232',  'c233',  'c234',  'c236',  'c237',  'c241',  'c242',  'c243',  'c244',  'c245',  'c246',  'c247',  'c248',  'c249',  'c250',  'c252',  'c253'],
    kept_channel_names = {
        # 153 channels
        'S1':  ['c13','c27','c30','c31','c32','c46','c47','c48','c49','c50','c51','c52','c53','c54','c55','c56','c57','c58','c59','c60','c61','c62','c63','c64','c65','c66','c67','c68','c70','c71','c72','c73','c74','c75','c76','c77','c78','c79','c80','c82','c83','c84','c85','c86','c87','c88','c89','c90','c91','c92','c94','c102','c106','c109','c111','c113','c114','c115','c116','c117','c118','c119','c120','c121','c123','c124','c125','c126','c127','c128','c129','c130','c131','c132','c133','c134','c135','c136','c137','c138','c139','c140','c141','c142','c143','c144','c161','c162','c163','c164','c165','c166','c167','c168','c169','c171','c172','c174','c175','c176','c177','c178','c179','c180','c181','c182','c183','c184','c185','c186','c187','c188','c190','c191','c192','c194','c196','c197','c199','c200','c201','c203','c204','c205','c206','c207','c208','c214','c218','c225','c226','c227','c228','c229','c230','c231','c232','c233','c234','c236','c237','c241','c242','c243','c244','c245','c246','c247','c248','c249','c250','c252','c253'],
    }
)