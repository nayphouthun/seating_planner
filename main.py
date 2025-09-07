import streamlit as st

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd
import random
import io

def reshape_to_table_seats(x):
        table_count = st.session_state.table_count
        guest_list = st.session_state.guest_list
        table_seats = x.reshape(table_count, len(guest_list))
        return table_seats

def cost(x):
        table_seats = reshape_to_table_seats(x)
        relationships_mat = st.session_state.relationships_mat
        table_costs = np.matrix(table_seats) * np.matrix(relationships_mat) * np.matrix(table_seats.T)
        table_cost = np.trace(table_costs)
        return table_cost

def take_step(x):
        table_seats = reshape_to_table_seats(np.matrix(x, copy=True))
        # randomly swap two guests
        table_count = st.session_state.table_count
        table_from, table_to = np.random.choice(table_count, 2, replace=False)
        
        table_from_guests = np.where(table_seats[table_from] == 1)[1]
        table_to_guests = np.where(table_seats[table_to] == 1)[1]
        
        table_from_guest = np.random.choice(table_from_guests)
        table_to_guest = np.random.choice(table_to_guests)
        
        table_seats[table_from, table_from_guest] = 0
        table_seats[table_from, table_to_guest] = 1
        table_seats[table_to, table_to_guest] = 0
        table_seats[table_to, table_from_guest] = 1
        return table_seats

def prob_accept(cost_old, cost_new, temp):
        a = 1 if cost_new < cost_old else np.exp((cost_old - cost_new) / temp)
        return a

def anneal(pos_current, temp=1.0, temp_min=0.00001, alpha=0.9, n_iter=100, audit=False):
        cost_old = cost(pos_current)
        
        audit_trail = []
        
        while temp > temp_min:
                for i in range(0, n_iter):
                        pos_new = take_step(pos_current)
                        cost_new = cost(pos_new)
                        p_accept = prob_accept(cost_old, cost_new, temp)
                        if p_accept > np.random.random():
                                pos_current = pos_new
                                cost_old = cost_new
                        if audit:
                                audit_trail.append((cost_new, cost_old, temp, p_accept))
                temp *= alpha
        
        return pos_current, cost_old, audit_trail

def main():
        st.header("Seating Plan Generator")
        st.write("Generate optimal seating arrangement while accounting for pairings to prioritise or avoid.")
        
        with st.expander("Instructions", expanded=True):
                with open('Seating_Plan_Input_Template.xlsx', 'rb') as f:
                        st.download_button('Download template input file', f, file_name='Seating_Plan_Input_Template.xlsx.xlsx')
                
                st.write('With the input template file, enter all attendees names in \'Guest\' column then the names of other guests they should be either paired with in the \'Together\' columns or avoid sitting with in the \'Apart\' columns. Names of guests to pair with or avoid should be written exactly the same as in the \'Guest\' column.')
     
        file = st.file_uploader(label='Upload input csv/xlsx file', type=['csv', 'xlsx', 'xls'], key='upload_file')
        if file is None:
                st.info("Please upload an input file to proceed.")

        else:
                try:
                        # Read the uploaded CSV file into a pandas DataFrame
                        GuestListRaw = pd.read_excel(file, 0)
                        st.success("File uploaded and read successfully!")

                        # Display the DataFrame
                        st.subheader("Imported File:")
                        st.dataframe(GuestListRaw)

                except Exception as e:
                        st.error(f"Error reading file: {e}")

                table_size = st.number_input("Seats per Table", value = 8, step = 1, min_value=2, key='table_size')

                n_days = st.number_input("Number of Days", value = 4, step = 1, min_value=1, key='n_days')
                avoid_same_table = st.checkbox("Avoid seating guests who already sat together on subsequent days", value=True, key="avoid_same_table")
                if st.button("Generate Seating Plan"):
                        st.session_state.all_results = []
                        for day in range(n_days):
                                if day > 0:
                                        prev_day_df = st.session_state.all_results[day-1]
                                        prev_day_seatings = dict(zip(prev_day_df['Guest'], prev_day_df['Assigned Table No']))
                                        # prev_day_df.groupby('Assigned Table No')['Guest'].apply(list).to_dict()
                                        # print(prev_day_seatings)
                                        
                                # GuestListRaw = pd.read_excel(InputFileName, 'GuestList')

                                guest_list=GuestListRaw["Guest"].values.tolist()

                                RelMatrixRaw=GuestListRaw.dropna(thresh=2)

                                # RelMatrixRaw
                                relationships_edges={}
                                Together1=RelMatrixRaw[["Guest","Together1"]].dropna(thresh=2)
                                Together2=RelMatrixRaw[["Guest","Together2"]].dropna(thresh=2)
                                Together3=RelMatrixRaw[["Guest","Together3"]].dropna(thresh=2)
                                Together1.columns=["Guest","Together"]
                                Together2.columns=["Guest","Together"]
                                Together3.columns=["Guest","Together"]
                                Together=pd.concat([Together1,Together2,Together3])
                                Apart1=RelMatrixRaw[["Guest","Apart1"]].dropna(thresh=2)
                                Apart2=RelMatrixRaw[["Guest","Apart2"]].dropna(thresh=2)
                                Apart3=RelMatrixRaw[["Guest","Apart3"]].dropna(thresh=2)
                                Apart1.columns=["Guest","Apart"]
                                Apart2.columns=["Guest","Apart"]
                                Apart3.columns=["Guest","Apart"]
                                Apart=pd.concat([Apart1,Apart2,Apart3])
                                for element in list(zip(Together["Guest"], Together["Together"])):
                                        relationships_edges.update({element:-100})
                                for element in list(zip(Apart["Guest"], Apart["Apart"])):
                                        relationships_edges.update({element:100})
                                
                                # st.write("Imported Seating Arrangement Constraints List")
                                # st.write(" ")
                                # st.write("Default Value For Together = -50 , Default Value For Apart = +50")
                                # st.write("Uncomment Following Cell To Modify/Add/Delete")
                                # st.write("E.g Can Increase Strength Of Relationship by Using 1-100 e.g. -100 very close friends, +100 very bitter enemies")
                                # print(relationships_edges)
                                
                                table_count = len(guest_list) // table_size
                                # print("No Of Guests: "+str(len(guest_list)))
                                if len(guest_list)%table_size==0:
                                        blank_seats=[]
                                if len(guest_list)%table_size!=0:
                                        blank_seats=list(range(1,((table_count+1)*table_size-len(guest_list))+1))
                                        blank_seats=["Spare-"+str(x) for x in blank_seats]
                                guest_list=guest_list+blank_seats
                                st.session_state.guest_list = guest_list
                                # len(guest_list)
                                table_count = len(guest_list) // table_size
                                # print("TABLE_COUNT",table_count)
                                st.session_state.table_count = table_count
                                st.session_state.guest_list = guest_list

                                # print("No Of Seats Per Table: "+str(table_size))
                                # print("Required No Of Tables: "+str(table_count))
                                # print("No Of Spare Seats: "+str(len(blank_seats)))
                                
                                #Generate Relationship Matrix
                                # print("DAY", day+1)          
                                if day == 0 or avoid_same_table == False:
                                        # print("FIRST DAY")
                                        temp_graph = nx.Graph()
                                        guests_w_constraints = []
                                        for k, v in relationships_edges.items():
                                                temp_graph.add_edge(k[0], k[1], weight=v)

                                                if k[0] not in guests_w_constraints:
                                                        guests_w_constraints.append(k[0])
                                                if k[1] not in guests_w_constraints:
                                                        guests_w_constraints.append(k[1])
                        
                                        guests_wo_constraints = [guest for guest in guest_list if guest not in guests_w_constraints]

                                        for element in list(zip(guests_wo_constraints, guests_wo_constraints)):
                                                relationships_edges.update({element:0})
                                        
                                        for k, v in relationships_edges.items():
                                                if k[0] in guests_wo_constraints and k[1] in guests_wo_constraints:
                                                        temp_graph.add_edge(k[0], k[1], weight=v)
                                                        
                                else:
                                        # print("SUBSEQUENT DAY")
                                        # If avoid seating people at same table again on subsequent days
                                        # st.session_state.prev_day_rels
                                        temp_graph = st.session_state.prev_day_graph
                                        for u,v,d in temp_graph.edges(data=True):
                                                u_table = prev_day_seatings[u]
                                                v_table = prev_day_seatings[v]
                                                if u_table == v_table and d['weight'] != -100:
                                                        d['weight']+=((100/n_days)-5)
                                
                                st.session_state.prev_day_graph = temp_graph
                                                        
                                relationships_mat_unnormed = nx.to_numpy_array(temp_graph.to_undirected(), nodelist = guest_list)
                                relationships_mat = relationships_mat_unnormed / 100
                                st.session_state.relationships_mat = relationships_mat
                                # print(relationships_mat)
                                
                                #CHECKS FOR INCONSISTENCIES BETWEEN RELATIONSHIPS 
                                # print(relationships_edges)

                                # pairs=[]
                                # rel_values=[]
                                # for pair, rel_value in relationships_edges.items():
                                #         pairs.append(pair)
                                #         rel_values.append(rel_value)

                                # pairs

                                # print("Checking For Pair Inconsistencies")
                                # print(" ")

                                # sorted(pairs[0])==sorted(pairs[0])
                                # rel_values[0]!=rel_values[0]

                                # indicesWithIssues=[]

                                # for i in range(0,len(pairs)):
                                #         for j in range(0,len(pairs)):
                                #                 if sorted(pairs[i])==sorted(pairs[j]) and rel_values[i]!=rel_values[j]:
                                #                         if i not in indicesWithIssues and j not in indicesWithIssues:
                                #                                 # print("Inconsistency Detected Between:")
                                #                                 # print(pairs[i],":",rel_values[i])
                                #                                 # print(pairs[j],":",rel_values[j])
                                #                                 # print(" ")
                                #                                 indicesWithIssues.append(i)
                                #                                 indicesWithIssues.append(j)            
                                #                 else:
                                #                         pass

                                # # print(" ")
                                # # print("Checking For Triad Inconsistencies")
                                # # print(" ")

                                # all_cliques= nx.enumerate_all_cliques(temp_graph)
                                # triad_cliques=[x for x in all_cliques if len(x)==3 ]
                                # checkSignForTriad=[]
                                # for triad in triad_cliques:
                                #         print("Identified A Triad Consisting Of :",triad)
                                #         for i in range(0,len(pairs)):
                                #                 if sorted(triad[1:])==sorted(pairs[i]) or sorted(triad[:2])==sorted(pairs[i]) or sorted([triad[0],triad[2]])==sorted(pairs[i]):
                                #                         print(pairs[i],":",rel_values[i])
                                #                         checkSignForTriad.append(rel_values[i])
                                #         if (checkSignForTriad[0]<0 and checkSignForTriad[1]<0 and checkSignForTriad[2]<0) or (checkSignForTriad[0]>=0 and checkSignForTriad[1]>=0 and checkSignForTriad[2]>=0):
                                #                 print("Triad OK!")
                                #         else:
                                #                 print("Triad Inconsistent ! Please Re-Check As All Triangles Should Be ALL Together or ALL Apart")
                                #                 indicesWithIssues.append(i)
                                #         checkSignForTriad=[]
                                        # print(" ")

                                # if indicesWithIssues!=[]:
                                #         print("Warning ! If Not Corrected, Errors May Arise")
                                # else:
                                #         print("OK! No Inconsistencies Found")
                                
                                # View Relationship Matrix
                                RelationshipMatrix=pd.DataFrame(relationships_mat)
                                RelationshipMatrix.index=guest_list
                                RelationshipMatrix.columns=guest_list
                                RelationshipMatrix = RelationshipMatrix[(RelationshipMatrix.T != 0).any()]
                                RelationshipMatrix = RelationshipMatrix.loc[:, (RelationshipMatrix != 0).any(axis=0)]
                                # RelationshipMatrix
                                # View Relationship Matrix
                                
                                #Generate An Initial (Seed) Random Seating Arrangement
                                s = list(range(table_count*table_size))
                                random.shuffle(s)
                                s = [ x+1 for x in s]

                                Table_Arrangement=pd.DataFrame(zip(guest_list,s),columns=["Guest Name","Assigned Seat No"])
                                Table_Arrangement["Assigned Table No"]=((Table_Arrangement["Assigned Seat No"]-1)//table_size)+1

                                # Table_Arrangement.sort_values(by=['Assigned Table No'])

                                for i in range(1,table_count+1):
                                        Table_Arrangement["Table No "+str(i)]=np.where(Table_Arrangement['Assigned Table No']!= i, 0, 1)
                                Table_Arrangement_Transpose=Table_Arrangement.T
                                Table_Arrangement_Transpose=Table_Arrangement_Transpose.tail(len(Table_Arrangement_Transpose)-3)
                                initial_random_arrangement=Table_Arrangement_Transpose.values
                                # Table_Arrangement[["Guest Name","Assigned Table No"]]
                                
                                initial_random_arrangement_costs = np.matrix(initial_random_arrangement) * relationships_mat * initial_random_arrangement.T
                                # print(initial_random_arrangement_costs)

                                # print(np.trace(initial_random_arrangement_costs))
                                

                                with st.spinner("Generating plan for day " + str(day+1) + "..."):
                                        result = anneal(initial_random_arrangement,temp=1.0, temp_min=0.00001, alpha=0.9, n_iter=100, audit=False)
                                        
                                # print("Cost Function Of Optimized Seating Arrangement:",cost(result[0]),"vs. Initial Seed Cost Function Value Of",np.trace(initial_random_arrangement_costs))
                                # print("NB: Lower Number = Better")
                                
                                multiplier_table=[]
                                for i in range(1,table_count+1):
                                        multiplier_table.append([i])
                                
                                suggested_arrangement=pd.DataFrame(np.array(result[0])*np.array(multiplier_table)).T
                                suggested_arrangement.columns=Table_Arrangement.columns[-table_count:]
                                suggested_arrangement["Assigned Table No"]=suggested_arrangement.sum(axis=1)
                                suggested_arrangement['Guest Name']=Table_Arrangement['Guest Name']
                                suggested_arrangement=suggested_arrangement[["Guest Name","Assigned Table No"]]
                                # display(suggested_arrangement)
                                # display("Ordered By Table No")
                                # suggested_arrangement_by_tableNo=suggested_arrangement.sort_values(by=['Assigned Table No'])
                                # suggested_arrangement_by_tableNo[["Assigned Table No","Guest Name"]]
                                # display(suggested_arrangement_by_tableNo)
                                
                                FinalSeatingArrangementDF=suggested_arrangement
                                FinalSeatingArrangementDF.columns=["Guest","Assigned Table No"]
                                FinalSeatingArrangementWConditions = pd.merge(FinalSeatingArrangementDF,RelMatrixRaw,on ='Guest',how='left')
                                FinalSeatingArrangementWConditions =FinalSeatingArrangementWConditions[["Guest","Assigned Table No","Together1","Together2","Together3","Apart1","Apart2","Apart3"]]
                                FinalSeatingDisplay = FinalSeatingArrangementWConditions.sort_values(by=['Assigned Table No'])
                                
                                st.session_state.all_results.append(FinalSeatingDisplay)
                                
                        st.header("Seating Plan")
                        
                        for i in range(0,len(st.session_state.all_results)):
                                st.subheader('Day ' + str(i+1) + ' Seating Plan')
                                st.dataframe(st.session_state.all_results[i])
                                st.divider()
                                
                        writer = pd.ExcelWriter('Seating_Plan_output.xlsx', engine='xlsxwriter')
                        for i, frame in enumerate(st.session_state.all_results):
                                frame.to_excel(writer, sheet_name = "Day " + str(i+1), index=False)
                                
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                                for i, frame in enumerate(st.session_state.all_results):
                                        frame.to_excel(writer, sheet_name = "Day " + str(i+1), index=False)

                        download = st.download_button(label='Download Seating Plan', data=buffer, file_name='Seating_Output.xlsx', on_click="ignore")
                                
main()