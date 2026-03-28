class_names = ['DP.135', 'I408', 'I423b', 'P.102', 'P.103a', 'P.103b', 'P.103c', 'P.104', 'P.106a', 'P.106b', 'P.107a', 'P.112', 'P.115', 'P.117', 'P.123a', 'P.123b', 'P.124a', 'P.124b', 'P.124c', 'P.125', 'P.127', 'P.128', 'P.130', 'P.131a', 'P.137', 'P.245a', 'R.301c', 'R.301d', 'R.301e', 'R.302a', 'R.302b', 'R.303', 'R.407a', 'R.409', 'R.425', 'R.434', 'S.509a', 'W.201a', 'W.201b', 'W.202a', 'W.202b', 'W.203b', 'W.203c', 'W.205a', 'W.205b', 'W.205d', 'W.207a', 'W.207b', 'W.207c', 'W.208', 'W.209', 'W.210', 'W.224', 'W.225', 'W.227', 'W.245a']
en_descriptions = {'End of all restrictions', 'Parking', 'Pedestrian crossing', 'No Entry', 'No Cars', 'No Right Turn for Cars', 'No Left Turn for Cars', 'No Motorcycles', 'No Trucks', 'Weight Limit Trucks', 'No Buses/Trucks', 'No Pedestrians', 'Weight Limit', 'Height Limit', 'No Left Turn', 'No Right Turn', 'No U-Turn', 'No U-Turn for Cars', 'No U-Turn & Left Turn', 'No Overtaking', 'Speed Limit', 'No Honking', 'No Stopping & Parking', 'No Parking', 'No Straight & Left Turn', 'Slow Down', 'Turn Left Only', 'Turn Right Only', 'Turn Left Only', 'Turn Right', 'Turn Left', 'Roundabout', 'One Way', 'U-Turn Area', 'Populated Area', 'Priority Road', 'Supplementary Sign', 'Dangerous curve left', 'Dangerous curve right', 'Multiple curves left', 'Multiple curves right', 'Narrow road left', 'Narrow road right', 'Intersection', 'Intersection', 'Intersection', 'Intersection non-priority', 'Intersection non-priority', 'Intersection non-priority', 'Intersection priority', 'Traffic lights', 'Railway crossing', 'Pedestrian crossing ahead', 'Children', 'Construction', 'Slow Down'}
CLASS_MAP = {}
name_en = {}
for idx, code in enumerate(class_names):
    desc = en_descriptions.get(code, code)
    if '(' in desc and ')' in desc:
        en_desc = desc.split('(')[1].split(')')[0].strip()
        full_name = en_desc
    else:
        full_name = code
    CLASS_MAP[idx] = {'name_en': full_name}
    name_en[idx] = full_name